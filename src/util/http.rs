//! Tiny plain-HTTP helpers for optional control-plane probes.
//!
//! This intentionally supports `http://` only. HTTPS in this project is kept
//! behind explicit features because rustls' production providers currently pull
//! native crypto backends.

use std::io;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::time::timeout;

const MAX_HTTP_RESPONSE_BYTES: usize = 64 * 1024;

#[allow(dead_code)]
pub async fn get_text(url: &str, request_timeout: Duration) -> io::Result<String> {
    let parsed = url::Url::parse(url).map_err(invalid_input)?;
    if parsed.scheme() != "http" {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            format!("unsupported URL scheme '{}'", parsed.scheme()),
        ));
    }

    let host = parsed
        .host_str()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "URL has no host"))?;
    let port = parsed
        .port_or_known_default()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "URL has no known port"))?;

    let mut path = parsed.path().to_string();
    if path.is_empty() {
        path.push('/');
    }
    if let Some(query) = parsed.query() {
        path.push('?');
        path.push_str(query);
    }

    let mut stream = timeout(request_timeout, TcpStream::connect((host, port)))
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "HTTP connect timed out"))??;

    let host_header = if port == 80 {
        host.to_string()
    } else {
        format!("{host}:{port}")
    };
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {host_header}\r\nUser-Agent: telemt/1\r\nConnection: close\r\nAccept: text/plain,*/*;q=0.1\r\n\r\n"
    );

    timeout(request_timeout, stream.write_all(request.as_bytes()))
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "HTTP request write timed out"))??;
    timeout(request_timeout, stream.flush())
        .await
        .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "HTTP request flush timed out"))??;

    let mut response = Vec::new();
    timeout(
        request_timeout,
        stream
            .take(MAX_HTTP_RESPONSE_BYTES as u64)
            .read_to_end(&mut response),
    )
    .await
    .map_err(|_| io::Error::new(io::ErrorKind::TimedOut, "HTTP response read timed out"))??;

    let header_end = find_header_end(&response)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "HTTP headers incomplete"))?;
    let headers = std::str::from_utf8(&response[..header_end])
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let status = parse_status(headers)?;
    if !(200..=299).contains(&status) {
        return Err(io::Error::other(format!("HTTP status {status}")));
    }

    Ok(String::from_utf8_lossy(&response[header_end + 4..]).into_owned())
}

fn invalid_input(error: url::ParseError) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, error)
}

fn find_header_end(bytes: &[u8]) -> Option<usize> {
    bytes.windows(4).position(|window| window == b"\r\n\r\n")
}

fn parse_status(headers: &str) -> io::Result<u16> {
    let status = headers
        .lines()
        .next()
        .and_then(|line| line.split_whitespace().nth(1))
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "HTTP status missing"))?;
    status
        .parse()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}
