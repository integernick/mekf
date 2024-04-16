#pragma once

#include "constants.h"

#include <Eigen/Core>
#include <cfloat>

/**
 * @addtogroup mathlib_module
 * @{ 
 */

/**
 * @brief   Global position as latitude, longitude, altitude coordinates.
 *
 * @note    All operations with vectors use small angle approximation.
 */
class LatLonAlt {
public:
    /// Default constructor, all fields are NAN.
    LatLonAlt() = default;

    /// Construct position from latitude [rad], longitude [rad], altitude [m].
    static LatLonAlt from_rad(const double &latitude_rad, const double &longitude_rad, const float &altitude) {
        return LatLonAlt(latitude_rad, longitude_rad, altitude);
    }

    /// Construct position from latitude [deg], longitude [deg], altitude [m].
    static LatLonAlt from_deg(const double &latitude_deg, const double &longitude_deg, const float &altitude) {
        return LatLonAlt(radians(latitude_deg), radians(longitude_deg), altitude);
    }

    LatLonAlt &operator=(const LatLonAlt &v) {
        _latitude_rad = v._latitude_rad;
        _longitude_rad = v._longitude_rad;
        _altitude = v._altitude;
        return *this;
    }

    /// Get latitude in [deg].
    double get_latitude_deg() const {
        return degrees(_latitude_rad);
    }

    /// Get latitude in [rad].
    const double &get_latitude_rad() const {
        return _latitude_rad;
    }

    /// Set latitude in [deg].
    void set_latitude_deg(const double &latitude_deg) {
        _latitude_rad = radians(latitude_deg);
    }

    /// Get longitude in [deg].
    double get_longitude_deg() const {
        return degrees(_longitude_rad);
    }

    /// Get longitude in [rad].
    const double &get_longitude_rad() const {
        return _longitude_rad;
    }

    /// Set longitude in [deg].
    void set_longitude_deg(const double &longitude_deg) {
        _longitude_rad = radians(longitude_deg);
    }

    /// Set latitude and longitude from another object.
    void set_lat_lon(const LatLonAlt &lat_lon_alt) {
        _latitude_rad = lat_lon_alt._latitude_rad;
        _longitude_rad = lat_lon_alt._longitude_rad;
    }

    /// Get altitude in [m].
    const float &get_altitude() const {
        return _altitude;
    }

    /// Set altitude in [m].
    void set_altitude(const float &altitude) {
        _altitude = altitude;
    }

    /// Add vector in NED [m] in place.
    const LatLonAlt &operator+=(const Eigen::Vector3f &v) {
        _latitude_rad += v(0) / RADIUS_OF_EARTH;
        _longitude_rad += v(1) / (cosf(static_cast<float>(_latitude_rad)) * RADIUS_OF_EARTH);
        _altitude -= v(2);
        return *this;
    }

    /// Add negated vector in NED [m] in place.
    const LatLonAlt &operator-=(const Eigen::Vector3f &v) {
        _latitude_rad -= v(0) / RADIUS_OF_EARTH;
        _longitude_rad -= v(1) / (cosf(static_cast<float>(_latitude_rad)) * RADIUS_OF_EARTH);
        _altitude += v(2);
        return *this;
    }

    /// Add vector in NED [m].
    LatLonAlt operator+(const Eigen::Vector3f &v) const {
        return LatLonAlt(
                _latitude_rad + v(0) / RADIUS_OF_EARTH,
                _longitude_rad + v(1) / (cosf(static_cast<float>(_latitude_rad)) * RADIUS_OF_EARTH),
                _altitude - v(2)
        );
    }

    /// Add negated vector in NED [m].
    LatLonAlt operator-(const Eigen::Vector3f &v) const {
        return LatLonAlt(
                _latitude_rad - v(0) / RADIUS_OF_EARTH,
                _longitude_rad - v(1) / (cosf(static_cast<float>(_latitude_rad)) * RADIUS_OF_EARTH),
                _altitude + v(2)
        );
    }

    /// Difference between two points as vector in NED [m].
    Eigen::Vector3f operator-(const LatLonAlt &p) const {
        return Eigen::Vector3f(
                static_cast<float>(_latitude_rad - p._latitude_rad) * RADIUS_OF_EARTH,
                static_cast<float>(_longitude_rad - p._longitude_rad) * cosf(static_cast<float>(_latitude_rad)) *
                RADIUS_OF_EARTH,
                p._altitude - _altitude
        );
    }

private:
    double _latitude_rad = NAN;        ///< Latitude [rad]
    double _longitude_rad = NAN;    ///< Longitude [rad]
    float _altitude = NAN;            ///< Altitude [m]

    /// Construct position from latitude [rad], longitude [rad], altitude [m].
    LatLonAlt(const double &latitude_rad, const double &longitude_rad, const float &altitude) :
            _latitude_rad(latitude_rad), _longitude_rad(longitude_rad), _altitude(altitude) {
    }
};

/**
 * @}
 */
