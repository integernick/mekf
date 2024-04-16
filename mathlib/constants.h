#pragma once

#include <Eigen/Geometry>

/**
 * @addtogroup mathlib_module
 * @{
 */

// Mathematical constants
#undef M_PI
#define M_PI			3.14159265358979323846	// PI
#undef M_PI_F
#define M_PI_F          3.14159265358979323846f	// PI
#define HALF_SQRT_2_F   0.7071067811865476f	    // sqrt(2) / 2

// Physical constants
constexpr float ONE_G_F = 9.80665f;
constexpr float RADIUS_OF_EARTH = 6371000.f;    /**< Radius of the Earth [m] */

// Cast everything to double to avoid rounding to float if -fsingle-precision-constant flag set
constexpr const double RAD_TO_DEG = static_cast<double>(180.0) / static_cast<double>(M_PI);
constexpr const float RAD_TO_DEG_F = static_cast<float>(RAD_TO_DEG);

/**
 * @brief   Convert degrees to radians.
 */
inline constexpr float radians(float degrees) {
    return degrees / RAD_TO_DEG_F;
}

/**
 * @brief   Convert degrees to radians.
 */
inline constexpr double radians(double degrees) {
    return degrees / RAD_TO_DEG;
}

/**
 * @brief   Convert degrees to radians.
 *
 * @param degrees   A vector of angles in [deg].
 * @return  A vector of angles in [rad].
 */
inline Eigen::Vector3f radians(const Eigen::Vector3f &degrees) {
    return degrees / RAD_TO_DEG;
}

/**
 * @brief   Convert radians to degrees.
 */
inline constexpr float degrees(float radians) {
    return radians * RAD_TO_DEG_F;
}

/**
 * @brief   Convert radians to degrees.
 */
inline constexpr double degrees(double radians) {
    return radians * RAD_TO_DEG;
}

/**
 * @brief   Convert radians to degrees.
 *
 * @param degrees   A vector of angles in [rad].
 * @return  A vector of angles in [deg].
 */
inline Eigen::Vector3f degrees(const Eigen::Vector3f &radians) {
    return radians * RAD_TO_DEG;
}

/**
 * @}
 */
