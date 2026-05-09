use std::sync::Arc;
#[cfg(feature = "https-control-plane")]
use std::time::Duration;

#[cfg(feature = "https-control-plane")]
use http_body_util::{BodyExt, Empty};
#[cfg(feature = "https-control-plane")]
use hyper::header::{CONNECTION, DATE, HOST, USER_AGENT};
#[cfg(feature = "https-control-plane")]
use hyper::{Method, Request};
#[cfg(feature = "https-control-plane")]
use hyper_util::rt::TokioIo;
#[cfg(feature = "https-control-plane")]
use rustls::pki_types::ServerName;
#[cfg(feature = "https-control-plane")]
use tokio::net::TcpStream;
#[cfg(feature = "https-control-plane")]
use tokio::time::timeout;
#[cfg(feature = "https-control-plane")]
use tokio_rustls::TlsConnector;
#[cfg(feature = "https-control-plane")]
use tracing::debug;

use crate::error::{ProxyError, Result};
#[cfg(feature = "https-control-plane")]
use crate::network::dns_overrides::resolve_socket_addr;
#[cfg(not(feature = "https-control-plane"))]
use crate::transport::UpstreamManager;
#[cfg(feature = "https-control-plane")]
use crate::transport::{UpstreamManager, UpstreamStream};

#[cfg(feature = "https-control-plane")]
const HTTP_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
#[cfg(feature = "https-control-plane")]
const HTTP_REQUEST_TIMEOUT: Duration = Duration::from_secs(15);

pub(crate) struct HttpsGetResponse {
    pub(crate) status: u16,
    pub(crate) date_header: Option<String>,
    pub(crate) body: Vec<u8>,
}

#[cfg(feature = "https-control-plane")]
fn install_tls_provider() {
    let _ = rustls_rustcrypto::provider().install_default();
}

#[cfg(feature = "https-control-plane")]
fn build_tls_client_config() -> Arc<rustls::ClientConfig> {
    install_tls_provider();
    let mut root_store = rustls::RootCertStore::empty();
    root_store.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    let provider = rustls_rustcrypto::provider();
    let config = rustls::ClientConfig::builder_with_provider(Arc::new(provider))
        .with_protocol_versions(&[&rustls::version::TLS13, &rustls::version::TLS12])
        .expect("HTTPS fetch rustls protocol versions must be valid")
        .with_root_certificates(root_store)
        .with_no_client_auth();
    Arc::new(config)
}

#[cfg(feature = "https-control-plane")]
fn extract_host_port_path(url: &str) -> Result<(String, u16, String)> {
    let parsed =
        url::Url::parse(url).map_err(|e| ProxyError::Proxy(format!("invalid URL '{url}': {e}")))?;
    if parsed.scheme() != "https" {
        return Err(ProxyError::Proxy(format!(
            "unsupported URL scheme '{}': only https is supported",
            parsed.scheme()
        )));
    }

    let host = parsed
        .host_str()
        .ok_or_else(|| ProxyError::Proxy(format!("URL has no host: {url}")))?
        .to_string();
    let port = parsed
        .port_or_known_default()
        .ok_or_else(|| ProxyError::Proxy(format!("URL has no known port: {url}")))?;

    let mut path = parsed.path().to_string();
    if path.is_empty() {
        path.push('/');
    }
    if let Some(query) = parsed.query() {
        path.push('?');
        path.push_str(query);
    }

    Ok((host, port, path))
}

#[cfg(feature = "https-control-plane")]
async fn resolve_target_addr(host: &str, port: u16) -> Result<std::net::SocketAddr> {
    if let Some(addr) = resolve_socket_addr(host, port) {
        return Ok(addr);
    }

    let addrs: Vec<std::net::SocketAddr> = tokio::net::lookup_host((host, port))
        .await
        .map_err(|e| ProxyError::Proxy(format!("DNS resolve failed for {host}:{port}: {e}")))?
        .collect();

    if let Some(addr) = addrs.iter().copied().find(|addr| addr.is_ipv4()) {
        return Ok(addr);
    }

    addrs
        .first()
        .copied()
        .ok_or_else(|| ProxyError::Proxy(format!("DNS returned no addresses for {host}:{port}")))
}

#[cfg(feature = "https-control-plane")]
async fn connect_https_transport(
    host: &str,
    port: u16,
    upstream: Option<Arc<UpstreamManager>>,
) -> Result<UpstreamStream> {
    if let Some(manager) = upstream {
        let target = resolve_target_addr(host, port).await?;
        return timeout(HTTP_CONNECT_TIMEOUT, manager.connect(target, None, None))
            .await
            .map_err(|_| ProxyError::Proxy(format!("upstream connect timeout for {host}:{port}")))?
            .map_err(|e| {
                ProxyError::Proxy(format!("upstream connect failed for {host}:{port}: {e}"))
            });
    }

    if let Some(addr) = resolve_socket_addr(host, port) {
        let stream = timeout(HTTP_CONNECT_TIMEOUT, TcpStream::connect(addr))
            .await
            .map_err(|_| ProxyError::Proxy(format!("connect timeout for {host}:{port}")))?
            .map_err(|e| ProxyError::Proxy(format!("connect failed for {host}:{port}: {e}")))?;
        return Ok(UpstreamStream::Tcp(stream));
    }

    let stream = timeout(HTTP_CONNECT_TIMEOUT, TcpStream::connect((host, port)))
        .await
        .map_err(|_| ProxyError::Proxy(format!("connect timeout for {host}:{port}")))?
        .map_err(|e| ProxyError::Proxy(format!("connect failed for {host}:{port}: {e}")))?;
    Ok(UpstreamStream::Tcp(stream))
}

#[cfg(feature = "https-control-plane")]
async fn https_get_via_reqwest(url: &str) -> Result<HttpsGetResponse> {
    install_tls_provider();
    let client = reqwest::Client::builder()
        .connect_timeout(HTTP_CONNECT_TIMEOUT)
        .timeout(HTTP_REQUEST_TIMEOUT)
        .no_proxy()
        .user_agent("telemt-middle-proxy/1")
        .build()
        .map_err(|e| ProxyError::Proxy(format!("build HTTPS client failed for {url}: {e}")))?;

    let response = client
        .get(url)
        .header("connection", "close")
        .send()
        .await
        .map_err(|e| ProxyError::Proxy(format!("HTTP request failed for {url}: {e}")))?;

    let status = response.status().as_u16();
    let date_header = response
        .headers()
        .get("date")
        .and_then(|value| value.to_str().ok())
        .map(|value| value.to_string());
    let body = response
        .bytes()
        .await
        .map_err(|e| ProxyError::Proxy(format!("HTTP body read failed for {url}: {e}")))?
        .to_vec();

    Ok(HttpsGetResponse {
        status,
        date_header,
        body,
    })
}

#[cfg(feature = "https-control-plane")]
pub(crate) async fn https_get(
    url: &str,
    upstream: Option<Arc<UpstreamManager>>,
) -> Result<HttpsGetResponse> {
    let (host, port, path_and_query) = extract_host_port_path(url)?;

    if upstream.is_none() && resolve_socket_addr(&host, port).is_none() {
        return https_get_via_reqwest(url).await;
    }

    let stream = connect_https_transport(&host, port, upstream).await?;

    let server_name = ServerName::try_from(host.clone())
        .map_err(|_| ProxyError::Proxy(format!("invalid TLS server name: {host}")))?;
    let connector = TlsConnector::from(build_tls_client_config());
    let tls_stream = timeout(HTTP_REQUEST_TIMEOUT, connector.connect(server_name, stream))
        .await
        .map_err(|_| ProxyError::Proxy(format!("TLS handshake timeout for {host}:{port}")))?
        .map_err(|e| ProxyError::Proxy(format!("TLS handshake failed for {host}:{port}: {e}")))?;

    let (mut sender, connection) = hyper::client::conn::http1::handshake(TokioIo::new(tls_stream))
        .await
        .map_err(|e| ProxyError::Proxy(format!("HTTP handshake failed for {host}:{port}: {e}")))?;

    tokio::spawn(async move {
        if let Err(e) = connection.await {
            debug!(error = %e, "HTTPS fetch connection task failed");
        }
    });

    let host_header = if port == 443 {
        host.clone()
    } else {
        format!("{host}:{port}")
    };

    let request = Request::builder()
        .method(Method::GET)
        .uri(path_and_query)
        .header(HOST, host_header)
        .header(USER_AGENT, "telemt-middle-proxy/1")
        .header(CONNECTION, "close")
        .body(Empty::<bytes::Bytes>::new())
        .map_err(|e| ProxyError::Proxy(format!("build HTTP request failed for {url}: {e}")))?;

    let response = timeout(HTTP_REQUEST_TIMEOUT, sender.send_request(request))
        .await
        .map_err(|_| ProxyError::Proxy(format!("HTTP request timeout for {url}")))?
        .map_err(|e| ProxyError::Proxy(format!("HTTP request failed for {url}: {e}")))?;

    let status = response.status().as_u16();
    let date_header = response
        .headers()
        .get(DATE)
        .and_then(|value| value.to_str().ok())
        .map(|value| value.to_string());

    let body = timeout(HTTP_REQUEST_TIMEOUT, response.into_body().collect())
        .await
        .map_err(|_| ProxyError::Proxy(format!("HTTP body read timeout for {url}")))?
        .map_err(|e| ProxyError::Proxy(format!("HTTP body read failed for {url}: {e}")))?
        .to_bytes()
        .to_vec();

    Ok(HttpsGetResponse {
        status,
        date_header,
        body,
    })
}

#[cfg(not(feature = "https-control-plane"))]
pub(crate) async fn https_get(
    url: &str,
    upstream: Option<Arc<UpstreamManager>>,
) -> Result<HttpsGetResponse> {
    let _ = upstream;
    Err(ProxyError::Proxy(format!(
        "HTTPS control-plane fetch is disabled in this build; use cached proxy-secret/proxy-config or enable the 'https-control-plane' feature for {url}"
    )))
}
