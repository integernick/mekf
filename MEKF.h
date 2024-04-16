#pragma once

#include "mathlib/LatLonAlt.h"
#include "mathlib/kalman_filter/KalmanFilter.h"

/**
 * @addtogroup mekf_module
 * @{
 */

inline static constexpr float sq(float v) { return v * v; }

struct Observation {
    float value;
    float var;
};

struct Observation3D {
    const Eigen::Vector3f &value;
    const float &var;
};

class MEKF {
public:
    struct Parameters {
        /* Gyro variance, [rad/s]^2  */
        float var_gyro = sq(0.01f);
        /* Gyro bias variance, [rad/s]^2 */
        float var_gyro_bias = sq(0.0005f);

        /* Accelerometer variance, [m/s2]^2 */
        float var_acc = sq(1.0f);
        /* Accelerometer bias variance, [m/s2]^2 */
        float var_acc_bias = sq(0.2f);

        /* Magnetometer bias variance, [rad]^2 */
        float var_mag_bias = sq(0.2f);

        /* Barometer bias variance, [m/s]^2 */
        float var_baro_bias = sq(1.0f);
    };

    explicit MEKF();

    /**
     * @brief   Get estimator parameters structure.
     */
    Parameters &parameters();

    /**
     * @brief   Acquire vehicle attitude quaternion from the estimator state.
     * @return  The estimated vehicle's attitude quaternion.
     */
    [[nodiscard]] Eigen::Quaternionf &get_q();

    /**
     * @brief   Acquire vehicle velocity in NED coordinate frame from the estimator state.
     * @return  The estimated vehicle velocity vector in NED coordinate frame.
     */
    [[nodiscard]] Eigen::Vector3f &get_velocity();

    /**
     * @brief   Acquire integrated geographic coordinates from the estimator.
     * @return  The integrated geographic coordinates.
     */
    [[nodiscard]] LatLonAlt &get_lla();

    /**
     * @brief   Acquire gyro bias vector from the estimator state.
     * @return  The estimated gyro bias vector.
     */
    [[nodiscard]] Eigen::Ref<const Eigen::Vector3f> get_gyro_bias() const;

    /**
     * @brief   Acquire accelerometer bias vector from the estimator state.
     * @return  The estimated accelerometer bias vector.
     */
    [[nodiscard]] Eigen::Ref<const Eigen::Vector3f> get_acc_bias() const;

    /**
     * @brief   Acquire magnetometer bias vector from the estimator state.
     * @return  The estimated magnetometer bias vector.
     */
    [[nodiscard]] Eigen::Ref<const Eigen::Vector3f> get_mag_bias() const;

    /**
     * @brief   Acquire barometric altitude from the estimator state.
     * @return  The estimated barometric altitude.
     */
    [[nodiscard]] float get_alt_baro() const;

    /**
     * @brief   Acquire attitude variance vector from the estimator.
     * @return  The estimated attitude variance vector.
     */
    [[nodiscard]] Eigen::Vector3f get_var_att() const;

    /**
     * @brief   Acquire accelerometer bias vector variance vector from the estimator.
     * @return  The estimated accelerometer bias vector variance.
     */
    [[nodiscard]] Eigen::Vector3f get_var_acc_bias() const;

    /**
     * @brief   Acquire horizontal position variance from the estimator.
     * @return  The estimated horizontal position variance.
     */
    [[nodiscard]] float get_var_h_pos() const;

    /**
     * @brief   Acquire vertical position variance from the estimator.
     * @return  The estimated horizontal position variance.
     */
    [[nodiscard]] float get_var_v_pos() const;

    /**
     * @brief   Acquire horizontal velocity variance from the estimator.
     * @return  The estimated horizontal position variance.
     */
    [[nodiscard]] float get_var_h_vel() const;

    /**
     * @brief   Acquire vertical velocity variance from the estimator.
     * @return  The estimated vertical position variance.
     */
    [[nodiscard]] float get_var_v_vel() const;

    /**
     * @brief   Acquire barometric altitude variance from the estimator.
     * @return  The estimated barometric altitude variance.
     */
    [[nodiscard]] float get_var_alt_baro() const;

    /**
     * @brief   Initialize MEKF, reset states & variances to default values.
     */
    void init();

    /**
     * @brief   Initialize vehicle attitude with a provided quaternion.
     *
     * @param q_init        The initial quaternion value.
     * @param var_init_att  The initial attitude variance.
     */
    void init_attitude(const Eigen::Quaternionf& q_init, float var_init_att);

    /**
     * @brief   Predict system state (see \ref Prediction).
     *
     * @param dt    Time delta in [s].
     * @param gyro_raw  The observed raw gyro vector.
     * @param accel_raw The observed raw accel vector calculated for the reference point.
     */
    void predict(float dt, const Eigen::Vector3f &gyro_raw, const Eigen::Vector3f &accel_raw);

    /**
     * @brief   Correct estimator state with an accelerometer observation.
     */
    void correct_accel();

    /**
     * @brief   Correct estimated gyro bias vector with a set gyro bias vector.
     *
     * @param gyro_bias_obsv    The set gyro bias vector.
     */
    void correct_gyro_bias(const Observation3D &gyro_bias_obsv);

    /**
     * @brief   Correct estimator state with a magnetometer measurement.
     *
     * @param mag_obsv  The magnetometer observation.
     * @param mag_decl  The magnetometer declination angle in [rad].
     */
    void correct_mag(const Observation3D &mag_obsv, float mag_decl_rad = 0.0f);

    /**
     * @brief   Correct estimator state with a set heading value.
     *
     * @param heading   The set heading observation.
     */
    void correct_heading(const Observation &heading);

    /**
     * @brief   Correct estimator state with a GNSS attitude vector.
     *
     * @param q_obsv_ts     The attitude quaternion calculated at the GNSS solution timestamp.
     * @param att_vec_obsv  The observed GNSS attitude vector.
     */
    void correct_att_vec(const Eigen::Quaternionf &q_obsv_ts, const Observation3D &att_vec_obsv);

    /**
     * @brief   Correct estimator state with an observed horizontal position error.
     *
     * @param pos_h_err_obsv    The observed observed horizontal position error.
     */
    void correct_horizontal_position(const Observation3D &pos_h_err_obsv);

    /**
     * @brief   Correct estimator state with an observed vertical position error.
     *
     * @param pos_v_err_obsv    The observed observed vertical position error.
     */
    void correct_vertical_position(const Observation &pos_v_err_obsv);

    /**
     * @brief   Correct barometric altitude with a barometer measurement.
     *
     * @param baro_obsv The barometer observation.
     */
    void correct_baro(const Observation &baro_obsv);

    /**
     * @brief   Correct estimator state with an observed horizontal velocity error.
     *
     * @param vel_h_err_obsv    The observed observed horizontal velocity error.
     */
    void correct_horizontal_velocity(const Observation3D &vel_h_err_obsv);

    /**
     * @brief   Correct estimator state with an observed vertical velocity error.
     *
     * @param vel_v_err_obsv    The observed observed vertical velocity error.
     */
    void correct_vertical_velocity(const Observation &vel_v_err_obsv);

    /**
     * @brief   Fold filtered error state back into full state estimates.
     * @note    This method must always be called after all the corrections are applied.
     */
    void update_aposteriori();

private:
    enum {
        X = 0,
        Y = 1,
        Z = 2,
    };

    enum StateIdx {
        // Multiplicative orientation error
        IDX_ALPHA_X = 0,
        IDX_ALPHA_Y = 1,
        IDX_ALPHA_Z = 2,
        // Velocity error
        IDX_DVX = 3,
        IDX_DVY = 4,
        IDX_DVZ = 5,
        // Position error
        IDX_DX = 6,
        IDX_DY = 7,
        IDX_DZ = 8,
        // Gyro bias
        IDX_GYRO_BIAS_X = 9,
        IDX_GYRO_BIAS_Y = 10,
        IDX_GYRO_BIAS_Z = 11,
        // Accel bias
        IDX_ACC_BIAS_X = 12,
        IDX_ACC_BIAS_Y = 13,
        IDX_ACC_BIAS_Z = 14,
        // Magnetometer bias
        IDX_MAG_BIAS_X = 15,
        IDX_MAG_BIAS_Y = 16,
        IDX_MAG_BIAS_Z = 17,
        // Barometer altitude
        IDX_ALT_BARO = 18,

        IDX_SIZE = 19
    };

    static inline Eigen::Matrix3f skew_symmetric(const Eigen::Vector3f &v);

    template <typename T>
    static inline T constrain(T value, T min, T max);

    /**
     * @brief   Update covariance matrix after attitude quaternion prediction step.
     */
    template<typename T>
    void predict_attitude_cov(T &&cov_x, float dt);

    /**
     * @brief   Update covariance matrix after velocity and position prediction step.
     */
    template<typename T>
    void predict_position_velocity_cov(T &&cov_x, float dt);

    /**
     * @brief   Update system noise after prediction step
     */
    void predict_add_process_noise(Eigen::Matrix<float, IDX_SIZE, IDX_SIZE> &P, float dt);

    /**
     * @brief   Limit covariance matrix values.
     */
    void predict_limit_cov();

    /**
     * @brief   Add variance to the attitude parameter covariance matrix
     *          (see \ref System_process_noise).
     *
     * @param v The noise variance to be added to the covariance matrix.
     */
    void add_attitude_variance(float v);

    static constexpr const char *_name = "mekf";

    static constexpr float VAR_INIT_GYRO_BIAS = sq(0.01f);
    static constexpr float VAR_INIT_ACC_BIAS = sq(0.02f);
    static constexpr float VAR_INIT_VEL = sq(100.0f);
    static constexpr float LIMIT_GYRO_BIAS = 0.02f;
    static constexpr float LIMIT_ACC_BIAS = 0.2f;
    static constexpr float VAR_LIMIT_ACC_BIAS = sq(0.1f);

    Parameters _params;

    Eigen::Vector3f _gyro;  ///< Measured angular velocity in body frame, recalculated to reference point via offset
    Eigen::Vector3f _accel; ///< Measured acceleration in body frame, recalculated to reference point via offset

    Eigen::Matrix3f DdvDalpha; ///< d(dv)/d(alpha) 3x3 Jacobian matrix

    // Integrated state
    LatLonAlt _lla_estimate;
    Eigen::Vector3f _vel_estimate{0, 0, 0};
    Eigen::Quaternionf _q_estimate{1, 0, 0, 0};

    KalmanFilter<IDX_SIZE> _kalman;
};

/**
 * @}
 */
