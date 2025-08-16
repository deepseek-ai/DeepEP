#pragma once

#ifdef ENABLE_DEBUG_LOGS
#define DEVICE_LOG_DEBUG(fmt, ...) printf("[DEBUG][%s] " fmt "\n", __func__, ##__VA_ARGS__)

#define _DEVICE_LOG_DEBUG_LANE_IMPL(lane, fmt, ...) do { \
    if (lane_id == (lane)) { \
        printf("[DEBUG][%s] " fmt "\n", __func__, ##__VA_ARGS__); \
    } \
} while(0)

#define DEVICE_LOG_DEBUG_LANE(lane, fmt, ...) _DEVICE_LOG_DEBUG_LANE_IMPL(lane, fmt, ##__VA_ARGS__)

#define DEVICE_LOG_DEBUG_LANE_SYNC(lane, fmt, ...) do { \
    _DEVICE_LOG_DEBUG_LANE_IMPL(lane, fmt, ##__VA_ARGS__); \
    __syncwarp(); \
} while(0)
#else
#define DEVICE_LOG_DEBUG(...)
#define DEVICE_LOG_DEBUG_LANE(...)
#define DEVICE_LOG_DEBUG_LANE_SYNC(...)
#endif
