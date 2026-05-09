//! Time Sync

use chrono::{DateTime, Utc};
use std::time::Duration;
#[cfg(feature = "https-control-plane")]
use tracing::warn;
use tracing::{debug, error};

#[allow(dead_code)]
const TIME_SYNC_URL: &str = "https://core.telegram.org/getProxySecret";
#[allow(dead_code)]
const MAX_TIME_SKEW_SECS: i64 = 30;

/// Time sync result
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct TimeSyncResult {
    pub server_time: DateTime<Utc>,
    pub local_time: DateTime<Utc>,
    pub skew_secs: i64,
    pub is_skewed: bool,
}

/// Check time synchronization with Telegram servers
#[allow(dead_code)]
pub async fn check_time_sync() -> Option<TimeSyncResult> {
    let _ = Duration::from_secs(10);
    let _ = TIME_SYNC_URL;
    debug!("Time sync HTTPS probe is disabled in the trimmed production build");
    None
}

#[cfg(feature = "https-control-plane")]
#[allow(dead_code)]
fn time_sync_result_from_date_header(date_str: &str) -> Option<TimeSyncResult> {
    let server_time = DateTime::parse_from_rfc2822(date_str)
        .ok()?
        .with_timezone(&Utc);

    let local_time = Utc::now();
    let skew_secs = (local_time - server_time).num_seconds();
    let is_skewed = skew_secs.abs() > MAX_TIME_SKEW_SECS;

    let result = TimeSyncResult {
        server_time,
        local_time,
        skew_secs,
        is_skewed,
    };

    if is_skewed {
        warn!(
            server = %server_time,
            local = %local_time,
            skew = skew_secs,
            "Time skew detected"
        );
    } else {
        debug!(skew = skew_secs, "Time sync OK");
    }

    Some(result)
}

/// Background time sync task
#[allow(dead_code)]
pub async fn time_sync_task(check_interval: Duration) -> ! {
    loop {
        if let Some(result) = check_time_sync().await
            && result.is_skewed
        {
            error!(
                "System clock is off by {} seconds. Please sync your clock.",
                result.skew_secs
            );
        }

        tokio::time::sleep(check_interval).await;
    }
}
