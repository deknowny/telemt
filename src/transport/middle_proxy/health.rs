#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use rand::RngExt;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::{debug, info, warn};

use crate::config::MeFloorMode;
use crate::crypto::SecureRandom;
use crate::network::IpFamily;

use super::MePool;
use super::pool::MeFamilyRuntimeState;

const JITTER_FRAC_NUM: u64 = 2; // jitter up to 50% of backoff
#[allow(dead_code)]
const MAX_CONCURRENT_PER_DC_DEFAULT: usize = 1;
const SHADOW_ROTATE_RETRY_SECS: u64 = 30;
const IDLE_REFRESH_TRIGGER_BASE_SECS: u64 = 30;
const IDLE_REFRESH_TRIGGER_JITTER_SECS: u64 = 15;
const IDLE_REFRESH_RETRY_SECS: u64 = 8;
const IDLE_REFRESH_SUCCESS_GUARD_SECS: u64 = 1;
const IDLE_REFRESH_MAX_PER_CYCLE: usize = 2;
const HEALTH_RECONNECT_BUDGET_PER_CORE: usize = 2;
const HEALTH_RECONNECT_BUDGET_PER_DC: usize = 1;
const HEALTH_RECONNECT_BUDGET_MIN: usize = 4;
const HEALTH_RECONNECT_BUDGET_MAX: usize = 128;
const FAMILY_SUPPRESS_FAIL_STREAK_THRESHOLD: u32 = 5;
const FAMILY_SUPPRESS_DURATION_SECS: u64 = 60;
const FAMILY_RECOVER_SUCCESS_STREAK_TARGET: u32 = 2;
const HEALTH_DRAIN_CLOSE_BUDGET_PER_CORE: usize = 16;
const HEALTH_DRAIN_CLOSE_BUDGET_MIN: usize = 16;
const HEALTH_DRAIN_CLOSE_BUDGET_MAX: usize = 256;
const HEALTH_DRAIN_TIMEOUT_ENFORCER_INTERVAL_SECS: u64 = 1;
const HEALTH_EXCESS_IDLE_DRAIN_BUDGET_PER_DC: usize = 2;

#[derive(Debug, Clone)]
struct DcFloorPlanEntry {
    dc: i32,
    endpoints: Vec<SocketAddr>,
    alive: usize,
    base_required: usize,
    min_required: usize,
    target_required: usize,
    max_required: usize,
    bound_clients: usize,
    has_bound_clients: bool,
    floor_capped: bool,
}

#[derive(Debug, Clone)]
struct FamilyFloorPlan {
    by_dc: HashMap<i32, DcFloorPlanEntry>,
    active_cap_configured_total: usize,
    active_cap_effective_total: usize,
    warm_cap_configured_total: usize,
    warm_cap_effective_total: usize,
    active_writers_current: usize,
    warm_writers_current: usize,
    target_writers_total: usize,
}

#[derive(Debug, Clone, Copy)]
struct AdaptiveFloorTargetHold {
    target_required: usize,
    expires_at: Instant,
}

#[derive(Debug)]
struct FamilyReconnectOutcome {
    key: (i32, IpFamily),
    dc: i32,
    family: IpFamily,
    required: usize,
    endpoint_count: usize,
}

pub async fn me_health_monitor(pool: Arc<MePool>, rng: Arc<SecureRandom>, _min_connections: usize) {
    let mut backoff: HashMap<(i32, IpFamily), u64> = HashMap::new();
    let mut next_attempt: HashMap<(i32, IpFamily), Instant> = HashMap::new();
    let mut inflight: HashMap<(i32, IpFamily), usize> = HashMap::new();
    let mut outage_backoff: HashMap<(i32, IpFamily), u64> = HashMap::new();
    let mut outage_next_attempt: HashMap<(i32, IpFamily), Instant> = HashMap::new();
    let mut single_endpoint_outage: HashSet<(i32, IpFamily)> = HashSet::new();
    let mut shadow_rotate_deadline: HashMap<(i32, IpFamily), Instant> = HashMap::new();
    let mut idle_refresh_next_attempt: HashMap<(i32, IpFamily), Instant> = HashMap::new();
    let mut floor_warn_next_allowed: HashMap<(i32, IpFamily), Instant> = HashMap::new();
    let mut adaptive_floor_target_hold: HashMap<(i32, IpFamily), AdaptiveFloorTargetHold> =
        HashMap::new();
    let mut drain_warn_next_allowed: HashMap<u64, Instant> = HashMap::new();
    let mut degraded_interval = true;
    loop {
        let interval = if degraded_interval {
            pool.health_interval_unhealthy()
        } else {
            pool.health_interval_healthy()
        };
        tokio::time::sleep(interval).await;
        pool.prune_closed_writers().await;
        pool.sweep_endpoint_quarantine().await;
        reap_draining_writers(&pool, &mut drain_warn_next_allowed).await;
        let v4_degraded = check_family(
            IpFamily::V4,
            &pool,
            &rng,
            &mut backoff,
            &mut next_attempt,
            &mut inflight,
            &mut outage_backoff,
            &mut outage_next_attempt,
            &mut single_endpoint_outage,
            &mut shadow_rotate_deadline,
            &mut idle_refresh_next_attempt,
            &mut floor_warn_next_allowed,
            &mut adaptive_floor_target_hold,
        )
        .await;
        let v6_degraded = check_family(
            IpFamily::V6,
            &pool,
            &rng,
            &mut backoff,
            &mut next_attempt,
            &mut inflight,
            &mut outage_backoff,
            &mut outage_next_attempt,
            &mut single_endpoint_outage,
            &mut shadow_rotate_deadline,
            &mut idle_refresh_next_attempt,
            &mut floor_warn_next_allowed,
            &mut adaptive_floor_target_hold,
        )
        .await;
        update_family_runtime_state(&pool, IpFamily::V4, v4_degraded);
        update_family_runtime_state(&pool, IpFamily::V6, v6_degraded);
        degraded_interval = v4_degraded || v6_degraded;
    }
}

pub async fn me_drain_timeout_enforcer(pool: Arc<MePool>) {
    let mut drain_warn_next_allowed: HashMap<u64, Instant> = HashMap::new();
    loop {
        tokio::time::sleep(Duration::from_secs(
            HEALTH_DRAIN_TIMEOUT_ENFORCER_INTERVAL_SECS,
        ))
        .await;
        reap_draining_writers(&pool, &mut drain_warn_next_allowed).await;
    }
}

pub(super) async fn reap_draining_writers(
    pool: &Arc<MePool>,
    warn_next_allowed: &mut HashMap<u64, Instant>,
) {
    let now_epoch_secs = MePool::now_epoch_secs();
    let now = Instant::now();
    let drain_ttl_secs = pool
        .drain_runtime
        .me_pool_drain_ttl_secs
        .load(std::sync::atomic::Ordering::Relaxed);
    let drain_threshold = pool
        .drain_runtime
        .me_pool_drain_threshold
        .load(std::sync::atomic::Ordering::Relaxed);
    let activity = pool.registry.writer_activity_snapshot().await;
    let mut draining_writers = Vec::<DrainingWriterSnapshot>::new();
    let mut empty_writer_ids = Vec::<u64>::new();
    let mut force_close_writer_ids = Vec::<u64>::new();
    let writers = pool.writers.read().await;
    for writer in writers.iter() {
        if !writer.draining.load(std::sync::atomic::Ordering::Relaxed) {
            continue;
        }
        if activity
            .bound_clients_by_writer
            .get(&writer.id)
            .copied()
            .unwrap_or(0)
            == 0
        {
            empty_writer_ids.push(writer.id);
            continue;
        }
        draining_writers.push(DrainingWriterSnapshot {
            id: writer.id,
            writer_dc: writer.writer_dc,
            addr: writer.addr,
            generation: writer.generation,
            created_at: writer.created_at,
            draining_started_at_epoch_secs: writer
                .draining_started_at_epoch_secs
                .load(std::sync::atomic::Ordering::Relaxed),
            drain_deadline_epoch_secs: writer
                .drain_deadline_epoch_secs
                .load(std::sync::atomic::Ordering::Relaxed),
            allow_drain_fallback: writer
                .allow_drain_fallback
                .load(std::sync::atomic::Ordering::Relaxed),
        });
    }
    drop(writers);

    let overflow = if drain_threshold > 0 && draining_writers.len() > drain_threshold as usize {
        draining_writers
            .len()
            .saturating_sub(drain_threshold as usize)
    } else {
        0
    };

    if overflow > 0 {
        draining_writers.sort_by(|left, right| {
            left.draining_started_at_epoch_secs
                .cmp(&right.draining_started_at_epoch_secs)
                .then_with(|| left.created_at.cmp(&right.created_at))
                .then_with(|| left.id.cmp(&right.id))
        });
        warn!(
            draining_writers = draining_writers.len(),
            me_pool_drain_threshold = drain_threshold,
            removing_writers = overflow,
            "ME draining writer threshold exceeded, force-closing oldest draining writers"
        );
        for writer in draining_writers.drain(..overflow) {
            force_close_writer_ids.push(writer.id);
        }
    }

    for writer in draining_writers {
        if drain_ttl_secs > 0
            && writer.draining_started_at_epoch_secs != 0
            && now_epoch_secs.saturating_sub(writer.draining_started_at_epoch_secs) > drain_ttl_secs
            && should_emit_writer_warn(
                warn_next_allowed,
                writer.id,
                now,
                pool.warn_rate_limit_duration(),
            )
        {
            warn!(
                writer_id = writer.id,
                writer_dc = writer.writer_dc,
                endpoint = %writer.addr,
                generation = writer.generation,
                drain_ttl_secs,
                force_close_secs = pool
                    .drain_runtime
                    .me_pool_force_close_secs
                    .load(std::sync::atomic::Ordering::Relaxed),
                allow_drain_fallback = writer.allow_drain_fallback,
                "ME draining writer remains non-empty past drain TTL"
            );
        }
        if writer.drain_deadline_epoch_secs != 0
            && now_epoch_secs >= writer.drain_deadline_epoch_secs
        {
            warn!(writer_id = writer.id, "Drain timeout, force-closing");
            force_close_writer_ids.push(writer.id);
        }
    }

    let close_budget = health_drain_close_budget();
    let requested_force_close = force_close_writer_ids.len();
    let requested_empty_close = empty_writer_ids.len();
    let requested_close_total = requested_force_close.saturating_add(requested_empty_close);
    let mut closed_writer_ids = HashSet::<u64>::new();
    let mut closed_total = 0usize;
    for writer_id in force_close_writer_ids {
        if closed_total >= close_budget {
            break;
        }
        if !closed_writer_ids.insert(writer_id) {
            continue;
        }
        pool.stats.increment_pool_force_close_total();
        pool.remove_writer_and_close_clients_with_reason(writer_id, "drain_timeout_force_close")
            .await;
        closed_total = closed_total.saturating_add(1);
    }
    for writer_id in empty_writer_ids {
        if closed_total >= close_budget {
            break;
        }
        if !closed_writer_ids.insert(writer_id) {
            continue;
        }
        pool.remove_writer_and_close_clients_with_reason(writer_id, "drain_empty_close")
            .await;
        closed_total = closed_total.saturating_add(1);
    }

    let pending_close_total = requested_close_total.saturating_sub(closed_total);
    if pending_close_total > 0 {
        warn!(
            close_budget,
            closed_total,
            pending_close_total,
            "ME draining close backlog deferred to next health cycle"
        );
    }

    // Keep warn cooldown state for draining writers still present in the pool;
    // drop state only once a writer is actually removed.
    let active_draining_writer_ids = {
        let writers = pool.writers.read().await;
        writers
            .iter()
            .filter(|writer| writer.draining.load(std::sync::atomic::Ordering::Relaxed))
            .map(|writer| writer.id)
            .collect::<HashSet<u64>>()
    };
    warn_next_allowed.retain(|writer_id, _| active_draining_writer_ids.contains(writer_id));
}

pub(super) fn health_drain_close_budget() -> usize {
    let cpu_cores = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    cpu_cores
        .saturating_mul(HEALTH_DRAIN_CLOSE_BUDGET_PER_CORE)
        .clamp(HEALTH_DRAIN_CLOSE_BUDGET_MIN, HEALTH_DRAIN_CLOSE_BUDGET_MAX)
}

#[derive(Debug, Clone)]
struct DrainingWriterSnapshot {
    id: u64,
    writer_dc: i32,
    addr: SocketAddr,
    generation: u64,
    created_at: Instant,
    draining_started_at_epoch_secs: u64,
    drain_deadline_epoch_secs: u64,
    allow_drain_fallback: bool,
}

fn should_emit_writer_warn(
    next_allowed: &mut HashMap<u64, Instant>,
    writer_id: u64,
    now: Instant,
    cooldown: Duration,
) -> bool {
    let Some(ready_at) = next_allowed.get(&writer_id).copied() else {
        next_allowed.insert(writer_id, now + cooldown);
        return true;
    };
    if now >= ready_at {
        next_allowed.insert(writer_id, now + cooldown);
        return true;
    }
    false
}

async fn check_family(
    family: IpFamily,
    pool: &Arc<MePool>,
    rng: &Arc<SecureRandom>,
    backoff: &mut HashMap<(i32, IpFamily), u64>,
    next_attempt: &mut HashMap<(i32, IpFamily), Instant>,
    inflight: &mut HashMap<(i32, IpFamily), usize>,
    outage_backoff: &mut HashMap<(i32, IpFamily), u64>,
    outage_next_attempt: &mut HashMap<(i32, IpFamily), Instant>,
    single_endpoint_outage: &mut HashSet<(i32, IpFamily)>,
    shadow_rotate_deadline: &mut HashMap<(i32, IpFamily), Instant>,
    idle_refresh_next_attempt: &mut HashMap<(i32, IpFamily), Instant>,
    floor_warn_next_allowed: &mut HashMap<(i32, IpFamily), Instant>,
    adaptive_floor_target_hold: &mut HashMap<(i32, IpFamily), AdaptiveFloorTargetHold>,
) -> bool {
    let enabled = match family {
        IpFamily::V4 => pool.decision.ipv4_me,
        IpFamily::V6 => pool.decision.ipv6_me,
    };
    if !enabled {
        return false;
    }

    let mut family_degraded = false;

    let mut dc_endpoints = HashMap::<i32, Vec<SocketAddr>>::new();
    let map_guard = match family {
        IpFamily::V4 => pool.proxy_map_v4.read().await,
        IpFamily::V6 => pool.proxy_map_v6.read().await,
    };
    for (dc, addrs) in map_guard.iter() {
        let entry = dc_endpoints.entry(*dc).or_default();
        for (ip, port) in addrs.iter().copied() {
            entry.push(SocketAddr::new(ip, port));
        }
    }
    drop(map_guard);
    for endpoints in dc_endpoints.values_mut() {
        endpoints.sort_unstable();
        endpoints.dedup();
    }
    let reconnect_budget = health_reconnect_budget(pool, dc_endpoints.len());
    let reconnect_sem = Arc::new(Semaphore::new(reconnect_budget));

    if pool.floor_mode() == MeFloorMode::Static {}

    let mut live_addr_counts = HashMap::<(i32, SocketAddr), usize>::new();
    let mut live_writer_ids_by_addr = HashMap::<(i32, SocketAddr), Vec<u64>>::new();
    for writer in pool
        .writers
        .read()
        .await
        .iter()
        .filter(|w| !w.draining.load(std::sync::atomic::Ordering::Relaxed))
    {
        if !matches!(
            super::pool::WriterContour::from_u8(
                writer.contour.load(std::sync::atomic::Ordering::Relaxed),
            ),
            super::pool::WriterContour::Active
        ) {
            continue;
        }
        let key = (writer.writer_dc, writer.addr);
        *live_addr_counts.entry(key).or_insert(0) += 1;
        live_writer_ids_by_addr
            .entry(key)
            .or_default()
            .push(writer.id);
    }
    let writer_idle_since = pool.registry.writer_idle_since_snapshot().await;
    let bound_clients_by_writer = pool
        .registry
        .writer_activity_snapshot()
        .await
        .bound_clients_by_writer;
    let floor_plan = build_family_floor_plan(
        pool,
        family,
        &dc_endpoints,
        &live_addr_counts,
        &live_writer_ids_by_addr,
        &bound_clients_by_writer,
        adaptive_floor_target_hold,
    )
    .await;
    pool.set_adaptive_floor_runtime_caps(
        floor_plan.active_cap_configured_total,
        floor_plan.active_cap_effective_total,
        floor_plan.warm_cap_configured_total,
        floor_plan.warm_cap_effective_total,
        floor_plan.target_writers_total,
        floor_plan.active_writers_current,
        floor_plan.warm_writers_current,
    );
    let live_writer_ids_by_addr = Arc::new(live_writer_ids_by_addr);
    let writer_idle_since = Arc::new(writer_idle_since);
    let bound_clients_by_writer = Arc::new(bound_clients_by_writer);
    let mut reconnect_set = JoinSet::<FamilyReconnectOutcome>::new();

    for (dc, endpoints) in dc_endpoints {
        if endpoints.is_empty() {
            continue;
        }
        let key = (dc, family);
        let required = floor_plan
            .by_dc
            .get(&dc)
            .map(|entry| entry.target_required)
            .unwrap_or_else(|| {
                pool.required_writers_for_dc_with_floor_mode(endpoints.len(), false)
            });
        let alive = endpoints
            .iter()
            .map(|addr| *live_addr_counts.get(&(dc, *addr)).unwrap_or(&0))
            .sum::<usize>();

        if endpoints.len() == 1 && pool.single_endpoint_outage_mode_enabled() && alive < required {
            family_degraded = true;
            if single_endpoint_outage.insert(key) {
                pool.stats.increment_me_single_endpoint_outage_enter_total();
                warn!(
                    dc = %dc,
                    ?family,
                    alive,
                    required,
                    endpoint_count = endpoints.len(),
                    "Single-endpoint DC outage detected"
                );
            }

            recover_single_endpoint_outage(
                pool,
                rng,
                key,
                endpoints[0],
                alive,
                required,
                outage_backoff,
                outage_next_attempt,
                &reconnect_sem,
            )
            .await;
            continue;
        }

        if single_endpoint_outage.remove(&key) {
            pool.stats.increment_me_single_endpoint_outage_exit_total();
            outage_backoff.remove(&key);
            outage_next_attempt.remove(&key);
            shadow_rotate_deadline.remove(&key);
            idle_refresh_next_attempt.remove(&key);
            info!(
                dc = %dc,
                ?family,
                alive,
                required,
                endpoint_count = endpoints.len(),
                "Single-endpoint DC outage recovered"
            );
        }

        if alive > required {
            let drained = drain_excess_idle_writers_for_dc(
                pool,
                dc,
                family,
                &endpoints,
                alive,
                required,
                live_writer_ids_by_addr.as_ref(),
                writer_idle_since.as_ref(),
                bound_clients_by_writer.as_ref(),
            )
            .await;
            if drained > 0 {
                continue;
            }
        }

        if alive >= required {
            maybe_refresh_idle_writer_for_dc(
                pool,
                rng,
                key,
                dc,
                family,
                &endpoints,
                alive,
                required,
                live_writer_ids_by_addr.as_ref(),
                writer_idle_since.as_ref(),
                bound_clients_by_writer.as_ref(),
                idle_refresh_next_attempt,
            )
            .await;
            maybe_rotate_single_endpoint_shadow(
                pool,
                rng,
                key,
                dc,
                family,
                &endpoints,
                alive,
                required,
                live_writer_ids_by_addr.as_ref(),
                bound_clients_by_writer.as_ref(),
                shadow_rotate_deadline,
            )
            .await;
            continue;
        }
        let missing = required - alive;
        family_degraded = true;

        let now = Instant::now();
        if reconnect_sem.available_permits() == 0 {
            let base_ms = pool.reconnect_runtime.me_reconnect_backoff_base.as_millis() as u64;
            let next_ms = (*backoff.get(&key).unwrap_or(&base_ms)).max(base_ms);
            let jitter = next_ms / JITTER_FRAC_NUM;
            let wait = Duration::from_millis(next_ms)
                + Duration::from_millis(rand::rng().random_range(0..=jitter.max(1)));
            next_attempt.insert(key, now + wait);
            debug!(
                dc = %dc,
                ?family,
                alive,
                required,
                endpoint_count = endpoints.len(),
                reconnect_budget,
                "Skipping reconnect due to per-tick health reconnect budget"
            );
            continue;
        }
        if let Some(ts) = next_attempt.get(&key)
            && now < *ts
        {
            continue;
        }

        let max_concurrent = pool
            .reconnect_runtime
            .me_reconnect_max_concurrent_per_dc
            .max(1) as usize;
        if *inflight.get(&key).unwrap_or(&0) >= max_concurrent {
            continue;
        }
        if pool
            .has_refill_inflight_for_dc_key(super::pool::RefillDcKey { dc, family })
            .await
        {
            debug!(
                dc = %dc,
                ?family,
                alive,
                required,
                endpoint_count = endpoints.len(),
                "Skipping health reconnect: immediate refill is already in flight for this DC group"
            );
            continue;
        }
        *inflight.entry(key).or_insert(0) += 1;
        let pool_for_reconnect = pool.clone();
        let rng_for_reconnect = rng.clone();
        let reconnect_sem_for_dc = reconnect_sem.clone();
        let endpoints_for_dc = endpoints.clone();
        let live_writer_ids_by_addr_for_dc = live_writer_ids_by_addr.clone();
        let writer_idle_since_for_dc = writer_idle_since.clone();
        let bound_clients_by_writer_for_dc = bound_clients_by_writer.clone();
        let active_cap_effective_total = floor_plan.active_cap_effective_total;
        reconnect_set.spawn(async move {
            let mut restored = 0usize;
            for _ in 0..missing {
                let Ok(reconnect_permit) = reconnect_sem_for_dc.clone().try_acquire_owned() else {
                    break;
                };
                if pool_for_reconnect.active_contour_writer_count_total().await
                    >= active_cap_effective_total
                {
                    let swapped = maybe_swap_idle_writer_for_cap(
                        &pool_for_reconnect,
                        &rng_for_reconnect,
                        dc,
                        family,
                        &endpoints_for_dc,
                        live_writer_ids_by_addr_for_dc.as_ref(),
                        writer_idle_since_for_dc.as_ref(),
                        bound_clients_by_writer_for_dc.as_ref(),
                    )
                    .await;
                    if swapped {
                        pool_for_reconnect
                            .stats
                            .increment_me_floor_swap_idle_total();
                        restored += 1;
                        continue;
                    }

                    let base_req = pool_for_reconnect
                        .required_writers_for_dc_with_floor_mode(endpoints_for_dc.len(), false);
                    if alive + restored >= base_req {
                        pool_for_reconnect
                            .stats
                            .increment_me_floor_cap_block_total();
                        pool_for_reconnect
                            .stats
                            .increment_me_floor_swap_idle_failed_total();
                        debug!(
                            dc = %dc,
                            ?family,
                            alive,
                            required,
                            active_cap_effective_total,
                            "Adaptive floor cap reached, reconnect attempt blocked"
                        );
                        break;
                    }
                }
                pool_for_reconnect.stats.increment_me_reconnect_attempt();
                let res = tokio::time::timeout(
                    pool_for_reconnect.reconnect_runtime.me_one_timeout,
                    pool_for_reconnect.connect_endpoints_round_robin(
                        dc,
                        &endpoints_for_dc,
                        rng_for_reconnect.as_ref(),
                    ),
                )
                .await;
                match res {
                    Ok(true) => {
                        restored += 1;
                        pool_for_reconnect.stats.increment_me_reconnect_success();
                    }
                    Ok(false) => {
                        debug!(dc = %dc, ?family, "ME round-robin reconnect failed")
                    }
                    Err(_) => {
                        debug!(dc = %dc, ?family, "ME reconnect timed out");
                    }
                }
                drop(reconnect_permit);
            }

            FamilyReconnectOutcome {
                key,
                dc,
                family,
                required,
                endpoint_count: endpoints_for_dc.len(),
            }
        });
    }

    while let Some(joined) = reconnect_set.join_next().await {
        let outcome = match joined {
            Ok(outcome) => outcome,
            Err(join_error) => {
                debug!(error = %join_error, "Health reconnect task failed");
                continue;
            }
        };
        let now = Instant::now();
        let now_alive = live_active_writers_for_dc_family(pool, outcome.dc, outcome.family).await;
        if now_alive >= outcome.required {
            info!(
                dc = %outcome.dc,
                family = ?outcome.family,
                alive = now_alive,
                required = outcome.required,
                endpoint_count = outcome.endpoint_count,
                "ME writer floor restored for DC"
            );
            backoff.insert(
                outcome.key,
                pool.reconnect_runtime.me_reconnect_backoff_base.as_millis() as u64,
            );
            let jitter = pool.reconnect_runtime.me_reconnect_backoff_base.as_millis() as u64
                / JITTER_FRAC_NUM;
            let wait = pool.reconnect_runtime.me_reconnect_backoff_base
                + Duration::from_millis(rand::rng().random_range(0..=jitter.max(1)));
            next_attempt.insert(outcome.key, now + wait);
        } else {
            let curr = *backoff
                .get(&outcome.key)
                .unwrap_or(&(pool.reconnect_runtime.me_reconnect_backoff_base.as_millis() as u64));
            let next_ms = (curr.saturating_mul(2))
                .min(pool.reconnect_runtime.me_reconnect_backoff_cap.as_millis() as u64);
            backoff.insert(outcome.key, next_ms);
            let jitter = next_ms / JITTER_FRAC_NUM;
            let wait = Duration::from_millis(next_ms)
                + Duration::from_millis(rand::rng().random_range(0..=jitter.max(1)));
            next_attempt.insert(outcome.key, now + wait);
            if pool.is_runtime_ready() {
                let warn_cooldown = pool.warn_rate_limit_duration();
                if should_emit_rate_limited_warn(
                    floor_warn_next_allowed,
                    outcome.key,
                    now,
                    warn_cooldown,
                ) {
                    warn!(
                        dc = %outcome.dc,
                        family = ?outcome.family,
                        alive = now_alive,
                        required = outcome.required,
                        endpoint_count = outcome.endpoint_count,
                        backoff_ms = next_ms,
                        "DC writer floor is below required level, scheduled reconnect"
                    );
                }
            } else {
                info!(
                    dc = %outcome.dc,
                    family = ?outcome.family,
                    alive = now_alive,
                    required = outcome.required,
                    endpoint_count = outcome.endpoint_count,
                    backoff_ms = next_ms,
                    "DC writer floor is below required level during startup, scheduled reconnect"
                );
            }
        }
        if let Some(v) = inflight.get_mut(&outcome.key) {
            *v = v.saturating_sub(1);
        }
    }

    family_degraded
}

fn health_reconnect_budget(pool: &Arc<MePool>, dc_groups: usize) -> usize {
    let cpu_cores = pool.adaptive_floor_effective_cpu_cores().max(1);
    let by_cpu = cpu_cores.saturating_mul(HEALTH_RECONNECT_BUDGET_PER_CORE);
    let by_dc = dc_groups.saturating_mul(HEALTH_RECONNECT_BUDGET_PER_DC);
    by_cpu
        .saturating_add(by_dc)
        .clamp(HEALTH_RECONNECT_BUDGET_MIN, HEALTH_RECONNECT_BUDGET_MAX)
}

fn update_family_runtime_state(pool: &Arc<MePool>, family: IpFamily, degraded: bool) {
    let now_epoch_secs = MePool::now_epoch_secs();
    let previous_state = pool.family_runtime_state(family);
    let mut state_since_epoch_secs = pool.family_runtime_state_since_epoch_secs(family);
    let previous_suppressed_until_epoch_secs = pool.family_suppressed_until_epoch_secs(family);
    let previous_fail_streak = pool.family_fail_streak(family);
    let previous_recover_success_streak = pool.family_recover_success_streak(family);

    let (next_state, suppressed_until_epoch_secs, fail_streak, recover_success_streak) =
        if previous_suppressed_until_epoch_secs > now_epoch_secs {
            let fail_streak = if degraded {
                previous_fail_streak.saturating_add(1)
            } else {
                previous_fail_streak
            };
            (
                MeFamilyRuntimeState::Suppressed,
                previous_suppressed_until_epoch_secs,
                fail_streak,
                0,
            )
        } else if degraded {
            let fail_streak = previous_fail_streak.saturating_add(1);
            if fail_streak >= FAMILY_SUPPRESS_FAIL_STREAK_THRESHOLD {
                (
                    MeFamilyRuntimeState::Suppressed,
                    now_epoch_secs.saturating_add(FAMILY_SUPPRESS_DURATION_SECS),
                    fail_streak,
                    0,
                )
            } else {
                (MeFamilyRuntimeState::Degraded, 0, fail_streak, 0)
            }
        } else if matches!(previous_state, MeFamilyRuntimeState::Healthy) {
            (MeFamilyRuntimeState::Healthy, 0, 0, 0)
        } else {
            let recover_success_streak = previous_recover_success_streak.saturating_add(1);
            if recover_success_streak >= FAMILY_RECOVER_SUCCESS_STREAK_TARGET {
                (MeFamilyRuntimeState::Healthy, 0, 0, 0)
            } else {
                (
                    MeFamilyRuntimeState::Recovering,
                    0,
                    0,
                    recover_success_streak,
                )
            }
        };

    if next_state != previous_state || state_since_epoch_secs == 0 {
        state_since_epoch_secs = now_epoch_secs;
    }
    pool.set_family_runtime_state(
        family,
        next_state,
        state_since_epoch_secs,
        suppressed_until_epoch_secs,
        fail_streak,
        recover_success_streak,
    );
}

fn should_emit_rate_limited_warn(
    next_allowed: &mut HashMap<(i32, IpFamily), Instant>,
    key: (i32, IpFamily),
    now: Instant,
    cooldown: Duration,
) -> bool {
    let Some(ready_at) = next_allowed.get(&key).copied() else {
        next_allowed.insert(key, now + cooldown);
        return true;
    };
    if now >= ready_at {
        next_allowed.insert(key, now + cooldown);
        return true;
    }
    false
}

async fn live_active_writers_for_dc_family(pool: &Arc<MePool>, dc: i32, family: IpFamily) -> usize {
    let writers = pool.writers.read().await;
    writers
        .iter()
        .filter(|writer| {
            if writer.draining.load(std::sync::atomic::Ordering::Relaxed) {
                return false;
            }
            if writer.writer_dc != dc {
                return false;
            }
            if !matches!(
                super::pool::WriterContour::from_u8(
                    writer.contour.load(std::sync::atomic::Ordering::Relaxed),
                ),
                super::pool::WriterContour::Active
            ) {
                return false;
            }
            match family {
                IpFamily::V4 => writer.addr.is_ipv4(),
                IpFamily::V6 => writer.addr.is_ipv6(),
            }
        })
        .count()
}

fn adaptive_floor_class_min(
    pool: &Arc<MePool>,
    endpoint_count: usize,
    base_required: usize,
) -> usize {
    if endpoint_count <= 1 {
        // A single Telegram endpoint has no alternate address to absorb a
        // writer flap. Keep the base shadow floor even while the DC looks idle;
        // otherwise health drains to one writer and immediately has to rebuild
        // the floor when traffic resumes, creating availability churn.
        base_required.max(1)
    } else {
        pool.adaptive_floor_min_writers_multi_endpoint()
            .min(base_required.max(1))
    }
}

fn adaptive_floor_class_max(
    pool: &Arc<MePool>,
    endpoint_count: usize,
    base_required: usize,
    cpu_cores: usize,
) -> usize {
    let extra_per_core = if endpoint_count <= 1 {
        pool.adaptive_floor_max_extra_single_per_core()
    } else {
        pool.adaptive_floor_max_extra_multi_per_core()
    };
    base_required.saturating_add(cpu_cores.saturating_mul(extra_per_core))
}

fn adaptive_floor_load_required(bound_clients: usize) -> usize {
    const TARGET_CLIENTS_PER_WRITER: usize = 64;

    if bound_clients == 0 {
        0
    } else {
        bound_clients.saturating_add(TARGET_CLIENTS_PER_WRITER - 1) / TARGET_CLIENTS_PER_WRITER
    }
}

fn list_writer_ids_for_endpoints(
    dc: i32,
    endpoints: &[SocketAddr],
    live_writer_ids_by_addr: &HashMap<(i32, SocketAddr), Vec<u64>>,
) -> Vec<u64> {
    let mut out = Vec::<u64>::new();
    for endpoint in endpoints {
        if let Some(ids) = live_writer_ids_by_addr.get(&(dc, *endpoint)) {
            out.extend(ids.iter().copied());
        }
    }
    out
}

async fn build_family_floor_plan(
    pool: &Arc<MePool>,
    family: IpFamily,
    dc_endpoints: &HashMap<i32, Vec<SocketAddr>>,
    live_addr_counts: &HashMap<(i32, SocketAddr), usize>,
    live_writer_ids_by_addr: &HashMap<(i32, SocketAddr), Vec<u64>>,
    bound_clients_by_writer: &HashMap<u64, usize>,
    adaptive_floor_target_hold: &mut HashMap<(i32, IpFamily), AdaptiveFloorTargetHold>,
) -> FamilyFloorPlan {
    let mut entries = Vec::<DcFloorPlanEntry>::new();
    let mut by_dc = HashMap::<i32, DcFloorPlanEntry>::new();
    let mut family_active_total = 0usize;
    let mut active_dc_keys = HashSet::<i32>::new();

    let floor_mode = pool.floor_mode();
    let is_adaptive = floor_mode == MeFloorMode::Adaptive;
    let now = Instant::now();
    let target_hold_grace = Duration::from_secs(
        pool.floor_runtime
            .me_adaptive_floor_recover_grace_secs
            .load(std::sync::atomic::Ordering::Relaxed),
    );
    let cpu_cores = pool.adaptive_floor_effective_cpu_cores().max(1);
    let (active_writers_current, warm_writers_current, _) =
        pool.non_draining_writer_counts_by_contour().await;

    for (dc, endpoints) in dc_endpoints {
        if endpoints.is_empty() {
            continue;
        }
        active_dc_keys.insert(*dc);
        let _key = (*dc, family);
        let base_required = pool.required_writers_for_dc(endpoints.len()).max(1);
        let min_required = if is_adaptive {
            adaptive_floor_class_min(pool, endpoints.len(), base_required)
        } else {
            base_required
        };
        let mut max_required = if is_adaptive {
            adaptive_floor_class_max(pool, endpoints.len(), base_required, cpu_cores)
        } else {
            base_required
        };
        if max_required < min_required {
            max_required = min_required;
        }
        let alive = endpoints
            .iter()
            .map(|endpoint| {
                live_addr_counts
                    .get(&(*dc, *endpoint))
                    .copied()
                    .unwrap_or(0)
            })
            .sum::<usize>();
        family_active_total = family_active_total.saturating_add(alive);
        let writer_ids = list_writer_ids_for_endpoints(*dc, endpoints, live_writer_ids_by_addr);
        let bound_clients = writer_ids
            .iter()
            .map(|writer_id| bound_clients_by_writer.get(writer_id).copied().unwrap_or(0))
            .sum::<usize>();
        let has_bound_clients = bound_clients > 0;
        // Idle adaptive DCs keep quorum coverage instead of expanding to every
        // advertised endpoint. This prevents health checks from turning a large
        // Telegram proxy-config snapshot into an unbounded writer fanout.
        let desired_raw = if is_adaptive && !has_bound_clients {
            min_required
        } else {
            base_required.max(adaptive_floor_load_required(bound_clients))
        };
        let mut target_required = desired_raw.clamp(min_required, max_required);

        if is_adaptive && !target_hold_grace.is_zero() {
            target_required = apply_adaptive_floor_target_hold(
                adaptive_floor_target_hold,
                family,
                *dc,
                target_required,
                min_required,
                max_required,
                now,
                target_hold_grace,
            );
        }

        entries.push(DcFloorPlanEntry {
            dc: *dc,
            endpoints: endpoints.clone(),
            alive,
            base_required,
            min_required,
            target_required,
            max_required,
            bound_clients,
            has_bound_clients,
            floor_capped: false,
        });
    }

    adaptive_floor_target_hold.retain(|(dc, held_family), hold| {
        *held_family != family || (active_dc_keys.contains(dc) && hold.expires_at > now)
    });

    if entries.is_empty() {
        let active_cap_configured_total = pool.adaptive_floor_active_cap_configured_total();
        let warm_cap_configured_total = pool.adaptive_floor_warm_cap_configured_total();
        return FamilyFloorPlan {
            by_dc,
            active_cap_configured_total,
            active_cap_effective_total: active_cap_configured_total,
            warm_cap_configured_total,
            warm_cap_effective_total: warm_cap_configured_total,
            active_writers_current,
            warm_writers_current,
            target_writers_total: 0,
        };
    }

    if !is_adaptive {
        let target_total = entries
            .iter()
            .map(|entry| entry.target_required)
            .sum::<usize>();
        let active_cap_configured_total = pool.adaptive_floor_active_cap_configured_total();
        let warm_cap_configured_total = pool.adaptive_floor_warm_cap_configured_total();
        for entry in entries {
            by_dc.insert(entry.dc, entry);
        }
        return FamilyFloorPlan {
            by_dc,
            active_cap_configured_total,
            active_cap_effective_total: active_cap_configured_total.max(target_total),
            warm_cap_configured_total,
            warm_cap_effective_total: warm_cap_configured_total,
            active_writers_current,
            warm_writers_current,
            target_writers_total: target_total,
        };
    }

    let active_cap_configured_total = pool.adaptive_floor_active_cap_configured_total();
    let warm_cap_configured_total = pool.adaptive_floor_warm_cap_configured_total();
    let other_active = active_writers_current.saturating_sub(family_active_total);
    let min_sum = entries
        .iter()
        .map(|entry| entry.min_required)
        .sum::<usize>();
    let mut target_sum = entries
        .iter()
        .map(|entry| entry.target_required)
        .sum::<usize>();
    let family_cap = active_cap_configured_total
        .saturating_sub(other_active)
        .max(min_sum);
    if target_sum > family_cap {
        entries.sort_by_key(|entry| {
            (
                entry.has_bound_clients,
                std::cmp::Reverse(entry.target_required.saturating_sub(entry.min_required)),
                std::cmp::Reverse(entry.alive),
                entry.dc.abs(),
                entry.dc,
                entry.endpoints.len(),
                entry.max_required,
            )
        });
        let mut changed = true;
        while target_sum > family_cap && changed {
            changed = false;
            for entry in &mut entries {
                if target_sum <= family_cap {
                    break;
                }
                if entry.target_required > entry.min_required {
                    entry.target_required -= 1;
                    entry.floor_capped = true;
                    target_sum -= 1;
                    changed = true;
                }
            }
        }
    }

    for entry in entries {
        if entry.target_required > entry.base_required {
            debug!(
                dc = %entry.dc,
                ?family,
                alive = entry.alive,
                bound_clients = entry.bound_clients,
                base_required = entry.base_required,
                target_required = entry.target_required,
                max_required = entry.max_required,
                "ME adaptive floor expanded for active client load"
            );
        }
        by_dc.insert(entry.dc, entry);
    }
    let active_cap_effective_total =
        active_cap_configured_total.max(other_active.saturating_add(min_sum));
    let target_writers_total = other_active.saturating_add(target_sum);
    FamilyFloorPlan {
        by_dc,
        active_cap_configured_total,
        active_cap_effective_total,
        warm_cap_configured_total,
        warm_cap_effective_total: warm_cap_configured_total,
        active_writers_current,
        warm_writers_current,
        target_writers_total,
    }
}

fn apply_adaptive_floor_target_hold(
    adaptive_floor_target_hold: &mut HashMap<(i32, IpFamily), AdaptiveFloorTargetHold>,
    family: IpFamily,
    dc: i32,
    target_required: usize,
    min_required: usize,
    max_required: usize,
    now: Instant,
    target_hold_grace: Duration,
) -> usize {
    let key = (dc, family);
    let next_expires_at = now + target_hold_grace;

    let Some(held) = adaptive_floor_target_hold.get_mut(&key) else {
        adaptive_floor_target_hold.insert(
            key,
            AdaptiveFloorTargetHold {
                target_required,
                expires_at: next_expires_at,
            },
        );
        return target_required;
    };

    if target_required >= held.target_required || held.expires_at <= now {
        held.target_required = target_required;
        held.expires_at = next_expires_at;
        return target_required;
    }

    let held_target = held.target_required.clamp(min_required, max_required);
    if held_target > target_required {
        debug!(
            dc = %dc,
            ?family,
            computed_target = target_required,
            held_target,
            min_required,
            max_required,
            grace_left_ms = held.expires_at.saturating_duration_since(now).as_millis(),
            "ME adaptive floor held target during recovery grace"
        );
        held_target
    } else {
        target_required
    }
}

async fn maybe_swap_idle_writer_for_cap(
    pool: &Arc<MePool>,
    rng: &Arc<SecureRandom>,
    dc: i32,
    family: IpFamily,
    endpoints: &[SocketAddr],
    live_writer_ids_by_addr: &HashMap<(i32, SocketAddr), Vec<u64>>,
    writer_idle_since: &HashMap<u64, u64>,
    bound_clients_by_writer: &HashMap<u64, usize>,
) -> bool {
    let now_epoch_secs = MePool::now_epoch_secs();
    let mut candidate: Option<(u64, SocketAddr, u64)> = None;
    for endpoint in endpoints {
        let Some(writer_ids) = live_writer_ids_by_addr.get(&(dc, *endpoint)) else {
            continue;
        };
        for writer_id in writer_ids {
            if bound_clients_by_writer.get(writer_id).copied().unwrap_or(0) > 0 {
                continue;
            }
            let Some(idle_since_epoch_secs) = writer_idle_since.get(writer_id).copied() else {
                continue;
            };
            let idle_age_secs = now_epoch_secs.saturating_sub(idle_since_epoch_secs);
            if candidate
                .as_ref()
                .map(|(_, _, age)| idle_age_secs > *age)
                .unwrap_or(true)
            {
                candidate = Some((*writer_id, *endpoint, idle_age_secs));
            }
        }
    }

    let Some((old_writer_id, endpoint, idle_age_secs)) = candidate else {
        return false;
    };

    let connected = match tokio::time::timeout(
        pool.reconnect_runtime.me_one_timeout,
        pool.connect_one_for_dc(endpoint, dc, rng.as_ref()),
    )
    .await
    {
        Ok(Ok(())) => true,
        Ok(Err(error)) => {
            debug!(
                dc = %dc,
                ?family,
                %endpoint,
                old_writer_id,
                idle_age_secs,
                %error,
                "Adaptive floor cap swap connect failed"
            );
            false
        }
        Err(_) => {
            debug!(
                dc = %dc,
                ?family,
                %endpoint,
                old_writer_id,
                idle_age_secs,
                "Adaptive floor cap swap connect timed out"
            );
            false
        }
    };
    if !connected {
        return false;
    }

    pool.mark_writer_draining_with_timeout(old_writer_id, pool.force_close_timeout(), false)
        .await;
    info!(
        dc = %dc,
        ?family,
        %endpoint,
        old_writer_id,
        idle_age_secs,
        "Adaptive floor cap swap: idle writer rotated"
    );
    true
}

async fn drain_excess_idle_writers_for_dc(
    pool: &Arc<MePool>,
    dc: i32,
    family: IpFamily,
    endpoints: &[SocketAddr],
    alive: usize,
    required: usize,
    live_writer_ids_by_addr: &HashMap<(i32, SocketAddr), Vec<u64>>,
    writer_idle_since: &HashMap<u64, u64>,
    bound_clients_by_writer: &HashMap<u64, usize>,
) -> usize {
    if alive <= required {
        return 0;
    }

    let now_epoch_secs = MePool::now_epoch_secs();
    let mut candidates = Vec::<(u64, u64)>::new();
    for endpoint in endpoints {
        let Some(writer_ids) = live_writer_ids_by_addr.get(&(dc, *endpoint)) else {
            continue;
        };
        for writer_id in writer_ids {
            if bound_clients_by_writer.get(writer_id).copied().unwrap_or(0) > 0 {
                continue;
            }
            let Some(idle_since_epoch_secs) = writer_idle_since.get(writer_id).copied() else {
                continue;
            };
            candidates.push((
                *writer_id,
                now_epoch_secs.saturating_sub(idle_since_epoch_secs),
            ));
        }
    }

    if candidates.is_empty() {
        return 0;
    }

    candidates.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0)));
    let drain_count = alive
        .saturating_sub(required)
        .min(HEALTH_EXCESS_IDLE_DRAIN_BUDGET_PER_DC)
        .min(candidates.len());
    for (writer_id, _) in candidates.iter().take(drain_count) {
        pool.mark_writer_draining_with_timeout(*writer_id, pool.force_close_timeout(), false)
            .await;
    }
    info!(
        dc = %dc,
        ?family,
        alive,
        required,
        drained = drain_count,
        "ME adaptive floor drained excess idle writers"
    );
    drain_count
}

async fn maybe_refresh_idle_writer_for_dc(
    pool: &Arc<MePool>,
    rng: &Arc<SecureRandom>,
    key: (i32, IpFamily),
    dc: i32,
    family: IpFamily,
    endpoints: &[SocketAddr],
    alive: usize,
    required: usize,
    live_writer_ids_by_addr: &HashMap<(i32, SocketAddr), Vec<u64>>,
    writer_idle_since: &HashMap<u64, u64>,
    bound_clients_by_writer: &HashMap<u64, usize>,
    idle_refresh_next_attempt: &mut HashMap<(i32, IpFamily), Instant>,
) {
    if !idle_writer_pre_refresh_enabled(pool) {
        return;
    }

    // Single-endpoint DCs are fragile under sustained load: every proactive
    // rotate creates extra TCP churn against the same Telegram address. Let
    // keepalive and passive replacement handle idle closes there; health will
    // still refill immediately if the floor drops.
    if endpoints.len() <= 1 {
        return;
    }

    if alive < required {
        return;
    }

    let now = Instant::now();
    if let Some(next) = idle_refresh_next_attempt.get(&key)
        && now < *next
    {
        return;
    }

    let now_epoch_secs = MePool::now_epoch_secs();
    let mut candidates: Vec<(u64, SocketAddr, u64, u64)> = Vec::new();
    for endpoint in endpoints {
        let Some(writer_ids) = live_writer_ids_by_addr.get(&(dc, *endpoint)) else {
            continue;
        };
        for writer_id in writer_ids {
            if bound_clients_by_writer.get(writer_id).copied().unwrap_or(0) > 0 {
                continue;
            }
            let Some(idle_since_epoch_secs) = writer_idle_since.get(writer_id).copied() else {
                continue;
            };
            let idle_age_secs = now_epoch_secs.saturating_sub(idle_since_epoch_secs);
            let threshold_secs = IDLE_REFRESH_TRIGGER_BASE_SECS
                + (*writer_id % (IDLE_REFRESH_TRIGGER_JITTER_SECS + 1));
            if idle_age_secs < threshold_secs {
                continue;
            }
            candidates.push((*writer_id, *endpoint, idle_age_secs, threshold_secs));
        }
    }

    if candidates.is_empty() {
        return;
    }

    candidates.sort_by_key(|(_, _, idle_age_secs, _)| std::cmp::Reverse(*idle_age_secs));

    let mut refreshed = 0usize;
    for (old_writer_id, endpoint, idle_age_secs, threshold_secs) in
        candidates.into_iter().take(IDLE_REFRESH_MAX_PER_CYCLE)
    {
        let rotate_ok = match tokio::time::timeout(
            pool.reconnect_runtime.me_one_timeout,
            pool.connect_one_for_dc(endpoint, dc, rng.as_ref()),
        )
        .await
        {
            Ok(Ok(())) => true,
            Ok(Err(error)) => {
                debug!(
                    dc = %dc,
                    ?family,
                    %endpoint,
                    old_writer_id,
                    idle_age_secs,
                    threshold_secs,
                    %error,
                    "Idle writer pre-refresh connect failed"
                );
                false
            }
            Err(_) => {
                debug!(
                    dc = %dc,
                    ?family,
                    %endpoint,
                    old_writer_id,
                    idle_age_secs,
                    threshold_secs,
                    "Idle writer pre-refresh connect timed out"
                );
                false
            }
        };

        if !rotate_ok {
            idle_refresh_next_attempt
                .insert(key, now + Duration::from_secs(IDLE_REFRESH_RETRY_SECS));
            return;
        }

        pool.mark_writer_draining_with_timeout(old_writer_id, pool.force_close_timeout(), false)
            .await;
        refreshed += 1;
        info!(
            dc = %dc,
            ?family,
            %endpoint,
            old_writer_id,
            idle_age_secs,
            threshold_secs,
            alive,
            required,
            refreshed,
            max_per_cycle = IDLE_REFRESH_MAX_PER_CYCLE,
            "Idle writer refreshed before upstream idle timeout"
        );
    }

    if refreshed == 0 {
        return;
    }

    idle_refresh_next_attempt.insert(
        key,
        now + Duration::from_secs(IDLE_REFRESH_SUCCESS_GUARD_SECS),
    );
}

fn idle_writer_pre_refresh_enabled(pool: &MePool) -> bool {
    // Telegram may still close empty ME writers at about 90s even when
    // RPC_PING/RPC_PONG keepalive succeeds. Pre-refresh only rotates writers
    // with no bound clients, so it can safely run alongside keepalive.
    pool.writer_lifecycle.me_keepalive_enabled
}

async fn recover_single_endpoint_outage(
    pool: &Arc<MePool>,
    rng: &Arc<SecureRandom>,
    key: (i32, IpFamily),
    endpoint: SocketAddr,
    alive: usize,
    required: usize,
    outage_backoff: &mut HashMap<(i32, IpFamily), u64>,
    outage_next_attempt: &mut HashMap<(i32, IpFamily), Instant>,
    reconnect_sem: &Arc<Semaphore>,
) {
    let now = Instant::now();
    if let Some(ts) = outage_next_attempt.get(&key)
        && now < *ts
    {
        return;
    }

    let (min_backoff_ms, max_backoff_ms) = pool.single_endpoint_outage_backoff_bounds_ms();
    let missing = required.saturating_sub(alive).max(1);
    let max_attempts = pool
        .reconnect_runtime
        .me_reconnect_max_concurrent_per_dc
        .max(1) as usize;
    let attempts = missing
        .min(max_attempts)
        .min(reconnect_sem.available_permits());

    if attempts == 0 {
        outage_next_attempt.insert(key, now + Duration::from_millis(min_backoff_ms.max(250)));
        debug!(
            dc = %key.0,
            family = ?key.1,
            %endpoint,
            alive,
            required,
            missing,
            "Single-endpoint outage reconnect deferred by health reconnect budget"
        );
        return;
    }

    let mut successes = 0usize;
    let mut attempted = 0usize;
    for _ in 0..attempts {
        let Ok(_reconnect_permit) = reconnect_sem.clone().try_acquire_owned() else {
            break;
        };
        attempted += 1;
        pool.stats.increment_me_reconnect_attempt();
        pool.stats
            .increment_me_single_endpoint_outage_reconnect_attempt_total();

        let bypass_quarantine = pool.single_endpoint_outage_disable_quarantine();
        let attempt_ok = if bypass_quarantine {
            pool.stats
                .increment_me_single_endpoint_quarantine_bypass_total();
            match tokio::time::timeout(
                pool.reconnect_runtime.me_one_timeout,
                pool.connect_one_for_dc(endpoint, key.0, rng.as_ref()),
            )
            .await
            {
                Ok(Ok(())) => true,
                Ok(Err(e)) => {
                    debug!(
                        dc = %key.0,
                        family = ?key.1,
                        %endpoint,
                        error = %e,
                        "Single-endpoint outage reconnect failed (quarantine bypass path)"
                    );
                    false
                }
                Err(_) => {
                    debug!(
                        dc = %key.0,
                        family = ?key.1,
                        %endpoint,
                        "Single-endpoint outage reconnect timed out (quarantine bypass path)"
                    );
                    false
                }
            }
        } else {
            let one_endpoint = [endpoint];
            match tokio::time::timeout(
                pool.reconnect_runtime.me_one_timeout,
                pool.connect_endpoints_round_robin(key.0, &one_endpoint, rng.as_ref()),
            )
            .await
            {
                Ok(ok) => ok,
                Err(_) => {
                    debug!(
                        dc = %key.0,
                        family = ?key.1,
                        %endpoint,
                        "Single-endpoint outage reconnect timed out"
                    );
                    false
                }
            }
        };

        if !attempt_ok {
            break;
        }

        successes += 1;
        pool.stats
            .increment_me_single_endpoint_outage_reconnect_success_total();
        pool.stats.increment_me_reconnect_success();
    }

    if successes > 0 {
        outage_backoff.insert(key, min_backoff_ms);
        let jitter = min_backoff_ms / JITTER_FRAC_NUM;
        let wait = Duration::from_millis(min_backoff_ms)
            + Duration::from_millis(rand::rng().random_range(0..=jitter.max(1)));
        outage_next_attempt.insert(key, now + wait);
        info!(
            dc = %key.0,
            family = ?key.1,
            %endpoint,
            alive,
            required,
            missing,
            attempted,
            successes,
            alive_after = alive.saturating_add(successes),
            backoff_ms = min_backoff_ms,
            "Single-endpoint outage reconnect succeeded"
        );
        return;
    }

    let current_ms = *outage_backoff.get(&key).unwrap_or(&min_backoff_ms);
    let next_ms = current_ms.saturating_mul(2).min(max_backoff_ms);
    outage_backoff.insert(key, next_ms);
    let jitter = next_ms / JITTER_FRAC_NUM;
    let wait = Duration::from_millis(next_ms)
        + Duration::from_millis(rand::rng().random_range(0..=jitter.max(1)));
    outage_next_attempt.insert(key, now + wait);
    warn!(
        dc = %key.0,
        family = ?key.1,
        %endpoint,
        required,
        backoff_ms = next_ms,
        "Single-endpoint outage reconnect scheduled"
    );
}

async fn maybe_rotate_single_endpoint_shadow(
    pool: &Arc<MePool>,
    rng: &Arc<SecureRandom>,
    key: (i32, IpFamily),
    dc: i32,
    family: IpFamily,
    endpoints: &[SocketAddr],
    alive: usize,
    required: usize,
    live_writer_ids_by_addr: &HashMap<(i32, SocketAddr), Vec<u64>>,
    bound_clients_by_writer: &HashMap<u64, usize>,
    shadow_rotate_deadline: &mut HashMap<(i32, IpFamily), Instant>,
) {
    if endpoints.len() != 1 || alive < required {
        return;
    }

    let endpoint = endpoints[0];
    let bound_clients = live_writer_ids_by_addr
        .get(&(dc, endpoint))
        .map(|writer_ids| {
            writer_ids
                .iter()
                .map(|writer_id| bound_clients_by_writer.get(writer_id).copied().unwrap_or(0))
                .sum::<usize>()
        })
        .unwrap_or(0);
    if bound_clients > 0 {
        return;
    }

    let Some(interval) = pool.single_endpoint_shadow_rotate_interval() else {
        return;
    };

    let now = Instant::now();
    if let Some(deadline) = shadow_rotate_deadline.get(&key)
        && now < *deadline
    {
        return;
    }

    if pool.is_endpoint_quarantined(endpoint).await {
        pool.stats
            .increment_me_single_endpoint_shadow_rotate_skipped_quarantine_total();
        shadow_rotate_deadline.insert(key, now + Duration::from_secs(SHADOW_ROTATE_RETRY_SECS));
        debug!(
            dc = %dc,
            ?family,
            %endpoint,
            "Single-endpoint shadow rotation skipped: endpoint is quarantined"
        );
        return;
    }

    let Some(writer_ids) = live_writer_ids_by_addr.get(&(dc, endpoint)) else {
        shadow_rotate_deadline.insert(key, now + Duration::from_secs(SHADOW_ROTATE_RETRY_SECS));
        return;
    };

    let mut candidate_writer_id = None;
    for writer_id in writer_ids {
        if bound_clients_by_writer.get(writer_id).copied().unwrap_or(0) == 0 {
            candidate_writer_id = Some(*writer_id);
            break;
        }
    }

    let Some(old_writer_id) = candidate_writer_id else {
        shadow_rotate_deadline.insert(key, now + Duration::from_secs(SHADOW_ROTATE_RETRY_SECS));
        debug!(
            dc = %dc,
            ?family,
            %endpoint,
            alive,
            required,
            "Single-endpoint shadow rotation skipped: no empty writer candidate"
        );
        return;
    };

    let rotate_ok = match tokio::time::timeout(
        pool.reconnect_runtime.me_one_timeout,
        pool.connect_one_for_dc(endpoint, dc, rng.as_ref()),
    )
    .await
    {
        Ok(Ok(())) => true,
        Ok(Err(e)) => {
            debug!(
                dc = %dc,
                ?family,
                %endpoint,
                error = %e,
                "Single-endpoint shadow rotation connect failed"
            );
            false
        }
        Err(_) => {
            debug!(
                dc = %dc,
                ?family,
                %endpoint,
                "Single-endpoint shadow rotation connect timed out"
            );
            false
        }
    };

    if !rotate_ok {
        shadow_rotate_deadline.insert(
            key,
            now + interval.min(Duration::from_secs(SHADOW_ROTATE_RETRY_SECS)),
        );
        return;
    }

    pool.mark_writer_draining_with_timeout(old_writer_id, pool.force_close_timeout(), false)
        .await;
    pool.stats
        .increment_me_single_endpoint_shadow_rotate_total();
    shadow_rotate_deadline.insert(key, now + interval);
    info!(
        dc = %dc,
        ?family,
        %endpoint,
        old_writer_id,
        rotate_every_secs = interval.as_secs(),
        "Single-endpoint shadow writer rotated"
    );
}

/// Last-resort safety net for draining writers stuck past their deadline.
///
/// Runs every `TICK_SECS` and force-closes any draining writer whose
/// `drain_deadline_epoch_secs` has been exceeded by more than a threshold.
///
/// Two thresholds:
///   - `SOFT_THRESHOLD_SECS` (60s): writers with no bound clients
///   - `HARD_THRESHOLD_SECS` (300s): writers WITH bound clients (unconditional)
///
/// Intentionally kept trivial and independent of pool config to minimise
/// the probability of panicking itself. Uses `SystemTime` directly
/// as a fallback clock source and timeouts on every lock acquisition
/// and writer removal so one stuck writer cannot block the rest.
pub async fn me_zombie_writer_watchdog(pool: Arc<MePool>) {
    use std::time::{SystemTime, UNIX_EPOCH};

    const TICK_SECS: u64 = 30;
    const SOFT_THRESHOLD_SECS: u64 = 60;
    const HARD_THRESHOLD_SECS: u64 = 300;
    const LOCK_TIMEOUT_SECS: u64 = 5;
    const REMOVE_TIMEOUT_SECS: u64 = 10;
    const HARD_DETACH_TIMEOUT_STREAK: u8 = 3;

    let mut removal_timeout_streak = HashMap::<u64, u8>::new();

    loop {
        tokio::time::sleep(Duration::from_secs(TICK_SECS)).await;

        let now = match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(d) => d.as_secs(),
            Err(_) => continue,
        };

        // Phase 1: collect zombie IDs under a short read-lock with timeout.
        let zombie_ids_with_meta: Vec<(u64, bool)> = {
            let Ok(ws) =
                tokio::time::timeout(Duration::from_secs(LOCK_TIMEOUT_SECS), pool.writers.read())
                    .await
            else {
                warn!("zombie_watchdog: writers read-lock timeout, skipping tick");
                continue;
            };
            ws.iter()
                .filter(|w| w.draining.load(std::sync::atomic::Ordering::Relaxed))
                .filter_map(|w| {
                    let deadline = w
                        .drain_deadline_epoch_secs
                        .load(std::sync::atomic::Ordering::Relaxed);
                    if deadline == 0 {
                        return None;
                    }
                    let overdue = now.saturating_sub(deadline);
                    if overdue == 0 {
                        return None;
                    }
                    let started = w
                        .draining_started_at_epoch_secs
                        .load(std::sync::atomic::Ordering::Relaxed);
                    let drain_age = now.saturating_sub(started);
                    if drain_age > HARD_THRESHOLD_SECS {
                        return Some((w.id, true));
                    }
                    if overdue > SOFT_THRESHOLD_SECS {
                        return Some((w.id, false));
                    }
                    None
                })
                .collect()
        };
        // read lock released here

        if zombie_ids_with_meta.is_empty() {
            removal_timeout_streak.clear();
            continue;
        }

        let mut active_zombie_ids = HashSet::<u64>::with_capacity(zombie_ids_with_meta.len());
        for (writer_id, _) in &zombie_ids_with_meta {
            active_zombie_ids.insert(*writer_id);
        }
        removal_timeout_streak.retain(|writer_id, _| active_zombie_ids.contains(writer_id));

        warn!(
            zombie_count = zombie_ids_with_meta.len(),
            soft_threshold_secs = SOFT_THRESHOLD_SECS,
            hard_threshold_secs = HARD_THRESHOLD_SECS,
            "Zombie draining writers detected by watchdog, force-closing"
        );

        // Phase 2: remove each writer individually with a timeout.
        // One stuck removal cannot block the rest.
        for (writer_id, had_clients) in &zombie_ids_with_meta {
            let result = tokio::time::timeout(
                Duration::from_secs(REMOVE_TIMEOUT_SECS),
                pool.remove_writer_and_close_clients_with_reason(
                    *writer_id,
                    "zombie_watchdog_force_close",
                ),
            )
            .await;
            match result {
                Ok(removed) => {
                    removal_timeout_streak.remove(writer_id);
                    if removed {
                        pool.stats.increment_pool_force_close_total();
                        info!(writer_id, had_clients, "Zombie writer removed by watchdog");
                    } else {
                        debug!(
                            writer_id,
                            had_clients,
                            "Zombie writer was already removed before watchdog cleanup"
                        );
                    }
                }
                Err(_) => {
                    let streak = removal_timeout_streak
                        .entry(*writer_id)
                        .and_modify(|value| *value = value.saturating_add(1))
                        .or_insert(1);
                    warn!(
                        writer_id,
                        had_clients,
                        timeout_streak = *streak,
                        "Zombie writer removal timed out"
                    );
                    if *streak < HARD_DETACH_TIMEOUT_STREAK {
                        continue;
                    }

                    let hard_detach = tokio::time::timeout(
                        Duration::from_secs(REMOVE_TIMEOUT_SECS),
                        pool.remove_draining_writer_hard_detach(*writer_id),
                    )
                    .await;
                    match hard_detach {
                        Ok(true) => {
                            removal_timeout_streak.remove(writer_id);
                            pool.stats.increment_pool_force_close_total();
                            info!(
                                writer_id,
                                had_clients, "Zombie writer hard-detached after repeated timeouts"
                            );
                        }
                        Ok(false) => {
                            removal_timeout_streak.remove(writer_id);
                            debug!(
                                writer_id,
                                had_clients,
                                "Zombie hard-detach skipped (writer already gone or no longer draining)"
                            );
                        }
                        Err(_) => {
                            warn!(
                                writer_id,
                                had_clients, "Zombie hard-detach timed out, will retry next tick"
                            );
                        }
                    }
                }
            }
        }
    }
}
#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};
    use std::time::{Duration, Instant};

    use tokio::sync::mpsc;
    use tokio_util::sync::CancellationToken;

    use super::{
        build_family_floor_plan, drain_excess_idle_writers_for_dc, idle_writer_pre_refresh_enabled,
        reap_draining_writers,
    };
    use crate::config::{GeneralConfig, MeRouteNoWriterMode, MeSocksKdfPolicy, MeWriterPickMode};
    use crate::crypto::SecureRandom;
    use crate::network::IpFamily;
    use crate::network::probe::NetworkDecision;
    use crate::stats::Stats;
    use crate::transport::middle_proxy::codec::WriterCommand;
    use crate::transport::middle_proxy::pool::{MePool, MeWriter, WriterContour};
    use crate::transport::middle_proxy::registry::ConnMeta;

    async fn make_pool(me_pool_drain_threshold: u64) -> Arc<MePool> {
        let general = GeneralConfig {
            me_pool_drain_threshold,
            ..GeneralConfig::default()
        };
        let mut proxy_map_v4 = HashMap::new();
        proxy_map_v4.insert(2, vec![(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 10)), 443)]);
        let decision = NetworkDecision {
            ipv4_me: true,
            ..NetworkDecision::default()
        };
        MePool::new(
            None,
            vec![1u8; 32],
            None,
            false,
            None,
            Vec::new(),
            1,
            None,
            12,
            1200,
            proxy_map_v4,
            HashMap::new(),
            None,
            decision,
            None,
            Arc::new(SecureRandom::new()),
            Arc::new(Stats::default()),
            general.me_keepalive_enabled,
            general.me_keepalive_interval_secs,
            general.me_keepalive_jitter_secs,
            general.me_keepalive_payload_random,
            general.rpc_proxy_req_every,
            general.me_warmup_stagger_enabled,
            general.me_warmup_step_delay_ms,
            general.me_warmup_step_jitter_ms,
            general.me_reconnect_max_concurrent_per_dc,
            general.me_reconnect_backoff_base_ms,
            general.me_reconnect_backoff_cap_ms,
            general.me_reconnect_fast_retry_count,
            general.me_single_endpoint_shadow_writers,
            general.me_single_endpoint_outage_mode_enabled,
            general.me_single_endpoint_outage_disable_quarantine,
            general.me_single_endpoint_outage_backoff_min_ms,
            general.me_single_endpoint_outage_backoff_max_ms,
            general.me_single_endpoint_shadow_rotate_every_secs,
            general.me_floor_mode,
            general.me_adaptive_floor_idle_secs,
            general.me_adaptive_floor_min_writers_single_endpoint,
            general.me_adaptive_floor_min_writers_multi_endpoint,
            general.me_adaptive_floor_recover_grace_secs,
            general.me_adaptive_floor_writers_per_core_total,
            general.me_adaptive_floor_cpu_cores_override,
            general.me_adaptive_floor_max_extra_writers_single_per_core,
            general.me_adaptive_floor_max_extra_writers_multi_per_core,
            general.me_adaptive_floor_max_active_writers_per_core,
            general.me_adaptive_floor_max_warm_writers_per_core,
            general.me_adaptive_floor_max_active_writers_global,
            general.me_adaptive_floor_max_warm_writers_global,
            general.hardswap,
            general.me_pool_drain_ttl_secs,
            general.me_instadrain,
            general.me_pool_drain_threshold,
            general.me_pool_drain_soft_evict_enabled,
            general.me_pool_drain_soft_evict_grace_secs,
            general.me_pool_drain_soft_evict_per_writer,
            general.me_pool_drain_soft_evict_budget_per_core,
            general.me_pool_drain_soft_evict_cooldown_ms,
            general.effective_me_pool_force_close_secs(),
            general.me_pool_min_fresh_ratio,
            general.me_hardswap_warmup_delay_min_ms,
            general.me_hardswap_warmup_delay_max_ms,
            general.me_hardswap_warmup_extra_passes,
            general.me_hardswap_warmup_pass_backoff_base_ms,
            general.me_bind_stale_mode,
            general.me_bind_stale_ttl_secs,
            general.me_secret_atomic_snapshot,
            general.me_deterministic_writer_sort,
            MeWriterPickMode::default(),
            general.me_writer_pick_sample_size,
            MeSocksKdfPolicy::default(),
            general.me_writer_cmd_channel_capacity,
            general.me_route_channel_capacity,
            general.me_route_backpressure_enabled,
            general.me_route_fairshare_enabled,
            general.me_route_backpressure_base_timeout_ms,
            general.me_route_backpressure_high_timeout_ms,
            general.me_route_backpressure_high_watermark_pct,
            general.me_reader_route_data_wait_ms,
            general.me_health_interval_ms_unhealthy,
            general.me_health_interval_ms_healthy,
            general.me_warn_rate_limit_ms,
            MeRouteNoWriterMode::default(),
            general.me_route_no_writer_wait_ms,
            general.me_route_hybrid_max_wait_ms,
            general.me_route_blocking_send_timeout_ms,
            general.me_route_inline_recovery_attempts,
            general.me_route_inline_recovery_wait_ms,
        )
    }

    async fn insert_draining_writer(
        pool: &Arc<MePool>,
        writer_id: u64,
        drain_started_at_epoch_secs: u64,
    ) -> u64 {
        let (conn_id, _rx) = pool.registry.register().await;
        let (tx, _writer_rx) = mpsc::channel::<WriterCommand>(8);
        let writer = MeWriter {
            id: writer_id,
            addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 4000 + writer_id as u16),
            source_ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
            writer_dc: 2,
            generation: 1,
            contour: Arc::new(AtomicU8::new(WriterContour::Draining.as_u8())),
            created_at: Instant::now() - Duration::from_secs(writer_id),
            tx: tx.clone(),
            cancel: CancellationToken::new(),
            degraded: Arc::new(AtomicBool::new(false)),
            rtt_ema_ms_x10: Arc::new(AtomicU32::new(0)),
            draining: Arc::new(AtomicBool::new(true)),
            draining_started_at_epoch_secs: Arc::new(AtomicU64::new(drain_started_at_epoch_secs)),
            drain_deadline_epoch_secs: Arc::new(AtomicU64::new(0)),
            allow_drain_fallback: Arc::new(AtomicBool::new(false)),
        };
        pool.writers.write().await.push(writer);
        pool.registry.register_writer(writer_id, tx).await;
        pool.conn_count.fetch_add(1, Ordering::Relaxed);
        assert!(
            pool.registry
                .bind_writer(
                    conn_id,
                    writer_id,
                    ConnMeta {
                        target_dc: 2,
                        client_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 6000),
                        our_addr: SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 443),
                        proto_flags: 0,
                    },
                )
                .await
        );
        conn_id
    }

    async fn insert_live_writer(pool: &Arc<MePool>, writer_id: u64, writer_dc: i32) {
        let (tx, _writer_rx) = mpsc::channel::<WriterCommand>(8);
        let writer = MeWriter {
            id: writer_id,
            addr: SocketAddr::new(
                IpAddr::V4(Ipv4Addr::new(
                    203,
                    0,
                    113,
                    (writer_id as u8).saturating_add(1),
                )),
                4000 + writer_id as u16,
            ),
            source_ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
            writer_dc,
            generation: 2,
            contour: Arc::new(AtomicU8::new(WriterContour::Active.as_u8())),
            created_at: Instant::now(),
            tx: tx.clone(),
            cancel: CancellationToken::new(),
            degraded: Arc::new(AtomicBool::new(false)),
            rtt_ema_ms_x10: Arc::new(AtomicU32::new(0)),
            draining: Arc::new(AtomicBool::new(false)),
            draining_started_at_epoch_secs: Arc::new(AtomicU64::new(0)),
            drain_deadline_epoch_secs: Arc::new(AtomicU64::new(0)),
            allow_drain_fallback: Arc::new(AtomicBool::new(false)),
        };
        pool.writers.write().await.push(writer);
        pool.registry.register_writer(writer_id, tx).await;
        pool.conn_count.fetch_add(1, Ordering::Relaxed);
    }

    #[tokio::test]
    async fn idle_writer_pre_refresh_runs_with_keepalive_enabled() {
        let pool = make_pool(0).await;

        assert!(idle_writer_pre_refresh_enabled(&pool));
    }

    #[tokio::test]
    async fn adaptive_floor_single_endpoint_keeps_shadow_floor_when_idle() {
        let pool = make_pool(0).await;
        let endpoints = vec![SocketAddr::new(
            IpAddr::V4(Ipv4Addr::new(203, 0, 113, 1)),
            443,
        )];
        let dc_endpoints = HashMap::from([(2, endpoints)]);
        let live_addr_counts = HashMap::new();
        let live_writer_ids_by_addr = HashMap::new();
        let bound_clients_by_writer = HashMap::new();
        let mut adaptive_floor_target_hold = HashMap::new();

        let plan = build_family_floor_plan(
            &pool,
            IpFamily::V4,
            &dc_endpoints,
            &live_addr_counts,
            &live_writer_ids_by_addr,
            &bound_clients_by_writer,
            &mut adaptive_floor_target_hold,
        )
        .await;

        let entry = plan.by_dc.get(&2).expect("dc floor entry");
        assert_eq!(entry.min_required, 3);
        assert_eq!(entry.target_required, 3);
        assert_eq!(plan.target_writers_total, 3);
    }

    #[tokio::test]
    async fn adaptive_floor_idle_dc_uses_minimum_target_not_endpoint_fanout() {
        let pool = make_pool(0).await;
        let endpoints = vec![
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 1)), 443),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 2)), 443),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 3)), 443),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 4)), 443),
        ];
        let dc_endpoints = HashMap::from([(2, endpoints)]);
        let live_addr_counts = HashMap::new();
        let live_writer_ids_by_addr = HashMap::new();
        let bound_clients_by_writer = HashMap::new();
        let mut adaptive_floor_target_hold = HashMap::new();

        let plan = build_family_floor_plan(
            &pool,
            IpFamily::V4,
            &dc_endpoints,
            &live_addr_counts,
            &live_writer_ids_by_addr,
            &bound_clients_by_writer,
            &mut adaptive_floor_target_hold,
        )
        .await;

        let entry = plan.by_dc.get(&2).expect("dc floor entry");
        assert_eq!(entry.min_required, 1);
        assert!(entry.max_required >= 4);
        assert_eq!(entry.target_required, 1);
        assert_eq!(plan.target_writers_total, 1);
    }

    #[tokio::test]
    async fn adaptive_floor_bound_dc_keeps_base_endpoint_target() {
        let pool = make_pool(0).await;
        let endpoint = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 1)), 443);
        let endpoints = vec![
            endpoint,
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 2)), 443),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 3)), 443),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 4)), 443),
        ];
        let dc_endpoints = HashMap::from([(2, endpoints)]);
        let live_addr_counts = HashMap::from([((2, endpoint), 1usize)]);
        let live_writer_ids_by_addr = HashMap::from([((2, endpoint), vec![42u64])]);
        let bound_clients_by_writer = HashMap::from([(42u64, 1usize)]);
        let mut adaptive_floor_target_hold = HashMap::new();

        let plan = build_family_floor_plan(
            &pool,
            IpFamily::V4,
            &dc_endpoints,
            &live_addr_counts,
            &live_writer_ids_by_addr,
            &bound_clients_by_writer,
            &mut adaptive_floor_target_hold,
        )
        .await;

        let entry = plan.by_dc.get(&2).expect("dc floor entry");
        assert_eq!(entry.min_required, 1);
        assert_eq!(entry.target_required, 4);
        assert_eq!(plan.target_writers_total, 4);
    }

    #[tokio::test]
    async fn adaptive_floor_holds_recent_high_target_during_recovery_grace() {
        let pool = make_pool(0).await;
        let endpoint = SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 1)), 443);
        let endpoints = vec![
            endpoint,
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 2)), 443),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 3)), 443),
            SocketAddr::new(IpAddr::V4(Ipv4Addr::new(203, 0, 113, 4)), 443),
        ];
        let dc_endpoints = HashMap::from([(2, endpoints)]);
        let live_addr_counts = HashMap::from([((2, endpoint), 1usize)]);
        let live_writer_ids_by_addr = HashMap::from([((2, endpoint), vec![42u64])]);
        let mut adaptive_floor_target_hold = HashMap::new();

        let busy_plan = build_family_floor_plan(
            &pool,
            IpFamily::V4,
            &dc_endpoints,
            &live_addr_counts,
            &live_writer_ids_by_addr,
            &HashMap::from([(42u64, 640usize)]),
            &mut adaptive_floor_target_hold,
        )
        .await;
        let busy_target = busy_plan.by_dc.get(&2).unwrap().target_required;
        assert!(busy_target > 4);

        let idle_plan = build_family_floor_plan(
            &pool,
            IpFamily::V4,
            &dc_endpoints,
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &mut adaptive_floor_target_hold,
        )
        .await;

        assert_eq!(
            idle_plan.by_dc.get(&2).unwrap().target_required,
            busy_target
        );
        assert_eq!(idle_plan.target_writers_total, busy_target);
    }

    #[tokio::test]
    async fn adaptive_floor_drains_excess_idle_writers_with_budget() {
        let pool = make_pool(0).await;
        insert_live_writer(&pool, 1, 2).await;
        insert_live_writer(&pool, 2, 2).await;
        insert_live_writer(&pool, 3, 2).await;
        pool.registry.mark_writer_idle(1).await;
        pool.registry.mark_writer_idle(2).await;
        pool.registry.mark_writer_idle(3).await;

        let writers = pool.writers.read().await;
        let endpoints = writers.iter().map(|writer| writer.addr).collect::<Vec<_>>();
        let live_writer_ids_by_addr = writers
            .iter()
            .map(|writer| ((writer.writer_dc, writer.addr), vec![writer.id]))
            .collect::<HashMap<_, _>>();
        drop(writers);
        let writer_idle_since = pool.registry.writer_idle_since_snapshot().await;
        let bound_clients_by_writer = HashMap::new();

        let drained = drain_excess_idle_writers_for_dc(
            &pool,
            2,
            IpFamily::V4,
            &endpoints,
            3,
            1,
            &live_writer_ids_by_addr,
            &writer_idle_since,
            &bound_clients_by_writer,
        )
        .await;

        assert_eq!(drained, 2);
        let draining = pool
            .writers
            .read()
            .await
            .iter()
            .filter(|writer| writer.draining.load(Ordering::Relaxed))
            .count();
        assert_eq!(draining, 2);
    }

    #[tokio::test]
    async fn reap_draining_writers_force_closes_oldest_over_threshold() {
        let pool = make_pool(2).await;
        insert_live_writer(&pool, 1, 2).await;
        let now_epoch_secs = MePool::now_epoch_secs();
        let conn_a = insert_draining_writer(&pool, 10, now_epoch_secs.saturating_sub(30)).await;
        let conn_b = insert_draining_writer(&pool, 20, now_epoch_secs.saturating_sub(20)).await;
        let conn_c = insert_draining_writer(&pool, 30, now_epoch_secs.saturating_sub(10)).await;
        let mut warn_next_allowed = HashMap::new();

        reap_draining_writers(&pool, &mut warn_next_allowed).await;

        let mut writer_ids: Vec<u64> = pool
            .writers
            .read()
            .await
            .iter()
            .map(|writer| writer.id)
            .collect();
        writer_ids.sort_unstable();
        assert_eq!(writer_ids, vec![1, 20, 30]);
        assert!(pool.registry.get_writer(conn_a).await.is_none());
        assert_eq!(
            pool.registry.get_writer(conn_b).await.unwrap().writer_id,
            20
        );
        assert_eq!(
            pool.registry.get_writer(conn_c).await.unwrap().writer_id,
            30
        );
    }

    #[tokio::test]
    async fn reap_draining_writers_force_closes_overflow_without_replacement() {
        let pool = make_pool(2).await;
        let now_epoch_secs = MePool::now_epoch_secs();
        let conn_a = insert_draining_writer(&pool, 10, now_epoch_secs.saturating_sub(30)).await;
        let conn_b = insert_draining_writer(&pool, 20, now_epoch_secs.saturating_sub(20)).await;
        let conn_c = insert_draining_writer(&pool, 30, now_epoch_secs.saturating_sub(10)).await;
        let mut warn_next_allowed = HashMap::new();

        reap_draining_writers(&pool, &mut warn_next_allowed).await;

        let mut writer_ids: Vec<u64> = pool
            .writers
            .read()
            .await
            .iter()
            .map(|writer| writer.id)
            .collect();
        writer_ids.sort_unstable();
        assert_eq!(writer_ids, vec![20, 30]);
        assert!(pool.registry.get_writer(conn_a).await.is_none());
        assert_eq!(
            pool.registry.get_writer(conn_b).await.unwrap().writer_id,
            20
        );
        assert_eq!(
            pool.registry.get_writer(conn_c).await.unwrap().writer_id,
            30
        );
    }

    #[tokio::test]
    async fn reap_draining_writers_keeps_timeout_only_behavior_when_threshold_disabled() {
        let pool = make_pool(0).await;
        let now_epoch_secs = MePool::now_epoch_secs();
        let conn_a = insert_draining_writer(&pool, 10, now_epoch_secs.saturating_sub(30)).await;
        let conn_b = insert_draining_writer(&pool, 20, now_epoch_secs.saturating_sub(20)).await;
        let conn_c = insert_draining_writer(&pool, 30, now_epoch_secs.saturating_sub(10)).await;
        let mut warn_next_allowed = HashMap::new();

        reap_draining_writers(&pool, &mut warn_next_allowed).await;

        let writer_ids: Vec<u64> = pool
            .writers
            .read()
            .await
            .iter()
            .map(|writer| writer.id)
            .collect();
        assert_eq!(writer_ids, vec![10, 20, 30]);
        assert_eq!(
            pool.registry.get_writer(conn_a).await.unwrap().writer_id,
            10
        );
        assert_eq!(
            pool.registry.get_writer(conn_b).await.unwrap().writer_id,
            20
        );
        assert_eq!(
            pool.registry.get_writer(conn_c).await.unwrap().writer_id,
            30
        );
    }
}
