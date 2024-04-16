#include "MEKF.h"

#include <cmath>

using namespace Eigen;

MEKF::MEKF() {
    init();
}

MEKF::Parameters &MEKF::parameters() {
    return _params;
}

Quaternionf &MEKF::get_q() {
    return _q_estimate;
}

Vector3f &MEKF::get_velocity() {
    return _vel_estimate;
}

LatLonAlt &MEKF::get_lla() {
    return _lla_estimate;
}

Ref<const Vector3f> MEKF::get_gyro_bias() const {
    return _kalman.x.block<3, 1>(IDX_GYRO_BIAS_X, 0);
}

Ref<const Vector3f> MEKF::get_acc_bias() const {
    return _kalman.x.block<3, 1>(IDX_ACC_BIAS_X, 0);
}

Ref<const Vector3f> MEKF::get_mag_bias() const {
    return _kalman.x.block<3, 1>(IDX_MAG_BIAS_X, 0);
}

float MEKF::get_alt_baro() const {
    return _kalman.x(IDX_ALT_BARO);
}

Vector3f MEKF::get_var_att() const {
    return {_kalman.P(IDX_ALPHA_X, IDX_ALPHA_X),
            _kalman.P(IDX_ALPHA_Y, IDX_ALPHA_Y),
            _kalman.P(IDX_ALPHA_Z, IDX_ALPHA_Z)};
}

Vector3f MEKF::get_var_acc_bias() const {
    return {_kalman.P(IDX_ACC_BIAS_X, IDX_ACC_BIAS_X),
            _kalman.P(IDX_ACC_BIAS_Y, IDX_ACC_BIAS_Y),
            _kalman.P(IDX_ACC_BIAS_Z, IDX_ACC_BIAS_Z)};
}

float MEKF::get_var_h_pos() const {
    return _kalman.P(IDX_DX, IDX_DX) + _kalman.P(IDX_DY, IDX_DY);
}

float MEKF::get_var_v_pos() const {
    return _kalman.P(IDX_DZ, IDX_DZ);
}

float MEKF::get_var_h_vel() const {
    return _kalman.P(IDX_DVX, IDX_DVX) + _kalman.P(IDX_DVY, IDX_DVY);
}

float MEKF::get_var_v_vel() const {
    return _kalman.P(IDX_DVZ, IDX_DVZ);
}

float MEKF::get_var_alt_baro() const {
    return _kalman.P(IDX_ALT_BARO, IDX_ALT_BARO);
}

void MEKF::init() {
    _kalman.reset(IDX_DVX, 0, VAR_INIT_VEL);
    _kalman.reset(IDX_DVY, 0, VAR_INIT_VEL);
    _kalman.reset(IDX_DVZ, 0, VAR_INIT_VEL);
    _kalman.reset(IDX_GYRO_BIAS_X, 0, VAR_INIT_GYRO_BIAS);
    _kalman.reset(IDX_GYRO_BIAS_Y, 0, VAR_INIT_GYRO_BIAS);
    _kalman.reset(IDX_GYRO_BIAS_Z, 0, VAR_INIT_GYRO_BIAS);
    _kalman.reset(IDX_ACC_BIAS_X, 0, VAR_INIT_ACC_BIAS);
    _kalman.reset(IDX_ACC_BIAS_Y, 0, VAR_INIT_ACC_BIAS);
    _kalman.reset(IDX_ACC_BIAS_Z, 0, VAR_INIT_ACC_BIAS);
}

void MEKF::init_attitude(const Eigen::Quaternionf& q_init, float var_init_att) {
    _q_estimate = q_init;

    // Reset all orientation covariances
    _kalman.P.row(IDX_ALPHA_X).setZero();
    _kalman.P.row(IDX_ALPHA_Y).setZero();
    _kalman.P.row(IDX_ALPHA_Z).setZero();
    _kalman.P.col(IDX_ALPHA_X).setZero();
    _kalman.P.col(IDX_ALPHA_Y).setZero();
    _kalman.P.col(IDX_ALPHA_Z).setZero();

    add_attitude_variance(var_init_att);
}

void MEKF::predict(const float dt, const Vector3f &gyro_raw, const Vector3f &accel_raw) {
    _gyro = gyro_raw - get_gyro_bias();
    _accel = accel_raw - get_acc_bias();

    // Integrate angular velocity through forming a quaternion derivative
    _q_estimate.x() += 0.5f * _gyro(X) * dt;
    _q_estimate.y() += 0.5f * _gyro(Y) * dt;
    _q_estimate.z() += 0.5f * _gyro(Z) * dt;
    _q_estimate.normalize();

    // Calculate Jacobian for the covariance matrix update
    DdvDalpha = -_q_estimate.matrix() * skew_symmetric(_accel * dt);

    _kalman.predict_cov(
            [&](auto &&cov_x) {
                predict_attitude_cov(cov_x, dt);
                predict_position_velocity_cov(cov_x, dt);
            },
            [&](auto &P) {
                predict_add_process_noise(P, dt);
            });

    predict_limit_cov();
}

void MEKF::correct_accel() {
    const Quaternionf &q = _q_estimate;

    // Jacobian matrix
    Matrix<float, 3, IDX_SIZE> H_acc;
    H_acc.setZero();

    // gravity_body = q.conjugate().toRotationMatrix() * Vector3f(0, 0, ONE_G_F)
    const Vector3f gravity_body{2 * ONE_G_F * (q.w() * q.y() - q.x() * q.z()),
                                2 * ONE_G_F * (q.w() * q.x() + q.y() * q.z()),
                                ONE_G_F * (1.0f - 2 * q.x() * q.x() - 2 * q.y() * q.y())};

    H_acc.block<3, 3>(IDX_ALPHA_X, IDX_ALPHA_X) = skew_symmetric(gravity_body);
    H_acc(0, IDX_ACC_BIAS_X) = 1.0f;
    H_acc(1, IDX_ACC_BIAS_Y) = 1.0f;
    H_acc(2, IDX_ACC_BIAS_Z) = 1.0f;

    const Vector3f err = _accel - gravity_body;
    const Matrix3f R_accel_obsv_cov = Matrix3f::Identity() * _params.var_acc;
    _kalman.correct_block_err(IDX_ALPHA_X, err, H_acc, R_accel_obsv_cov);
}

void MEKF::correct_gyro_bias(const Observation3D &gyro_bias_obsv) {
    for (int i = 0; i < 3; ++i) {
        _kalman.correct(IDX_GYRO_BIAS_X + i, gyro_bias_obsv.value(i), gyro_bias_obsv.var);
    }
}

void MEKF::correct_mag(const Observation3D &mag_obsv, float mag_decl_rad) {
    const Quaternionf &q = _q_estimate;

    // Estimate is north vector (1, 0, 0) in the vehicle body frame
    // m_body = R_i^b * (1, 0, 0) = q.conjugate() * (1, 0, 0)
    // With decl: m_body = R_i^b * (Rz(-decl) * (1, 0, 0)) = q.conjugate() * Rz(-decl) * (1, 0, 0)
    /*const Vector3f mag_body_est{2 * q.w() * q.w() + 2 * q.x() * q.x(),
                                2 * q.x() * q.y() - 2 * q.w() * q.z(),
                                2 * q.w() * q.y() + 2 * q.x() * q.z()};*/
    const Vector3f declined_north = AngleAxisf(-mag_decl_rad, Vector3f::UnitZ()) * Vector3f::UnitX();
    const Vector3f mag_body_est = q.conjugate() * declined_north;

    // Jacobian matrix
    Matrix<float, 3, IDX_SIZE> H_mag;
    H_mag.setZero();
    H_mag.block<3, 3>(IDX_ALPHA_X, IDX_ALPHA_X) = skew_symmetric(mag_body_est);
    H_mag(0, IDX_MAG_BIAS_X) = 1.0f;
    H_mag(1, IDX_MAG_BIAS_Y) = 1.0f;
    H_mag(2, IDX_MAG_BIAS_Z) = 1.0f;

    const Vector3f err = mag_obsv.value - mag_body_est;
    const Matrix3f R_mag = Matrix3f::Identity() * mag_obsv.var;
    _kalman.correct_block_err(IDX_ALPHA_X, err, H_mag, R_mag);
}

void MEKF::correct_heading(const Observation &heading) {
    // Known heading on ground
    const Quaternionf &q = _q_estimate;
    const Vector3f heading_vec(cosf(heading.value), -sinf(heading.value), 0);
    // Estimate is north vector (1, 0, 0) in the vehicle body frame
    // m_body = R_i^b * (1, 0, 0)
    const Vector3f north_body_est{2 * q.w() * q.w() + 2 * q.x() * q.x(),
                                  2 * q.x() * q.y() - 2 * q.w() * q.z(),
                                  2 * q.w() * q.y() + 2 * q.x() * q.z()};
    // Jacobian matrix
    const Matrix<float, 3, 3> H_heading = skew_symmetric(north_body_est);

    const Vector3f err = heading_vec - north_body_est;
    const Matrix3f R_heading = Matrix3f::Identity() * heading.var;
    _kalman.correct_block_err(IDX_ALPHA_X, err, H_heading, R_heading);

}

void MEKF::correct_att_vec(const Quaternionf &q_obsv_ts, const Observation3D &att_vec_obsv) {
    // Apply attitude correction by GNSS attitude vector
    const Quaternionf &q = q_obsv_ts;
    // Estimate is vehicle's "y_body" in the inertial frame
    // att_vec_est = R_b^i * (0, 1, 0)
    const Vector3f att_vec_est{2 * q.x() * q.y() - 2 * q.w() * q.z(),
                               1.0f - 2 * q.x() * q.x() - 2 * q.z() * q.z(),
                               2 * q.w() * q.x() + 2 * q.y() * q.z()};
    const Vector3f err = att_vec_obsv.value - att_vec_est;

    // Jacobian matrix: [ -R_b^i(\hat{q}) * (0, 1, 0) \cross ]
    const Matrix<float, 3, 3> H_gnss_att = -skew_symmetric(att_vec_est);
    const Matrix<float, 3, 3> R = Matrix3f::Identity() * att_vec_obsv.var;
    _kalman.correct_block_err(IDX_ALPHA_X, err, H_gnss_att, R);
}

void MEKF::correct_horizontal_position(const Observation3D &pos_h_err_obsv) {
    const float var_xy = 0.5f * pos_h_err_obsv.var;
    _kalman.correct(IDX_DX, pos_h_err_obsv.value(0), var_xy);
    _kalman.correct(IDX_DY, pos_h_err_obsv.value(1), var_xy);
}

void MEKF::correct_vertical_position(const Observation &pos_v_err_obsv) {
    _kalman.correct(IDX_DZ, pos_v_err_obsv.value, pos_v_err_obsv.var);
}

void MEKF::correct_baro(const Observation &baro_obsv) {
    _kalman.correct(IDX_ALT_BARO, baro_obsv.value, baro_obsv.var);
}

void MEKF::correct_horizontal_velocity(const Observation3D &vel_h_err_obsv) {
    const float var_xy = 0.5f * vel_h_err_obsv.var;
    _kalman.correct(IDX_DVX, vel_h_err_obsv.value(0), var_xy);
    _kalman.correct(IDX_DVY, vel_h_err_obsv.value(1), var_xy);
}

void MEKF::correct_vertical_velocity(const Observation &vel_v_err_obsv) {
    _kalman.correct(IDX_DVZ, vel_v_err_obsv.value, vel_v_err_obsv.var);
}

template<typename T>
void MEKF::predict_attitude_cov(T &&cov_x, const float dt) {
    Map<Vector3f> a(&_kalman.x(IDX_ALPHA_X));

    cov_x(IDX_ALPHA_X) += dt * (/* 0 * cov_x(IDX_ALPHA_X) */ + _gyro(Z) * cov_x(IDX_ALPHA_Y) - _gyro(Y) * cov_x(IDX_ALPHA_Z)
                                - 1 * cov_x(IDX_GYRO_BIAS_X) + a.z() * cov_x(IDX_GYRO_BIAS_Y) - a.y() * cov_x(IDX_GYRO_BIAS_Z));
    cov_x(IDX_ALPHA_Y) += dt * (-_gyro(Z) * cov_x(IDX_ALPHA_X) /* + 0 * cov_x(IDX_ALPHA_Y) */ + _gyro(X) * cov_x(IDX_ALPHA_Z)
                                - a.z() * cov_x(IDX_GYRO_BIAS_X) - 1 * cov_x(IDX_GYRO_BIAS_Y) + a.x() * cov_x(IDX_GYRO_BIAS_Z));
    cov_x(IDX_ALPHA_Z) += dt * (_gyro(Y) * cov_x(IDX_ALPHA_X) - _gyro(X) * cov_x(IDX_ALPHA_Y) /* + 0 * cov_x(IDX_ALPHA_Z) */
                                + a.y() * cov_x(IDX_GYRO_BIAS_X) - a.x() * cov_x(IDX_GYRO_BIAS_Y) - 1 * cov_x(IDX_GYRO_BIAS_Z));
}

template<typename T>
void MEKF::predict_position_velocity_cov(T &&cov_x, const float dt) {
    const float half_dt = 0.5f * dt;

    const Vector3f d_cov_dv = DdvDalpha * Vector3f(cov_x(IDX_ALPHA_X), cov_x(IDX_ALPHA_Y), cov_x(IDX_ALPHA_Z));
    cov_x(IDX_DVX) += d_cov_dv(X);
    cov_x(IDX_DVY) += d_cov_dv(Y);
    cov_x(IDX_DVZ) += d_cov_dv(Z);

    // Accelerometer Z bias
    Vector3f acc_bias_body(cov_x(IDX_ACC_BIAS_X), cov_x(IDX_ACC_BIAS_Y), cov_x(IDX_ACC_BIAS_Z));
    Vector3f acc_bias_ned = _q_estimate * acc_bias_body;

    // Velocity and position covariance prediction
    for (int axis = 0; axis < 3; ++axis) {
        float dv = -acc_bias_ned(axis) * dt;
        float dp = cov_x(IDX_DVX + axis) * dt + dv * half_dt;
        if (axis == 2) {
            cov_x(IDX_ALT_BARO) -= dp;
        }
        cov_x(IDX_DVX + axis) += dv;
        cov_x(IDX_DX + axis) += dp;
    }
}

void MEKF::predict_add_process_noise(Matrix<float, IDX_SIZE, IDX_SIZE> &P, const float dt) {
    const float dt2_2 = dt * dt * 0.5f;
    const float dt3_3 = dt * dt * dt * 0.3333f;

    // alpha row
    const float var_alpha = _params.var_gyro * dt + _params.var_gyro_bias * dt3_3;
    add_attitude_variance(var_alpha);

    const float covar_alpha_gb = -_params.var_gyro_bias * dt2_2;
    P(IDX_ALPHA_X, IDX_GYRO_BIAS_X) += covar_alpha_gb;
    P(IDX_ALPHA_Y, IDX_GYRO_BIAS_Y) += covar_alpha_gb;
    P(IDX_ALPHA_Z, IDX_GYRO_BIAS_Z) += covar_alpha_gb;

    // dv row
    const float var_dv = _params.var_acc * dt + _params.var_acc_bias * dt3_3;
    P(IDX_DVX, IDX_DVX) += var_dv;
    P(IDX_DVY, IDX_DVY) += var_dv;
    P(IDX_DVZ, IDX_DVZ) += var_dv;

    const float covar_dv_dp = _params.var_acc * dt2_2;
    P(IDX_DVX, IDX_DX) += covar_dv_dp;
    P(IDX_DVY, IDX_DY) += covar_dv_dp;
    P(IDX_DVZ, IDX_DZ) += covar_dv_dp;

    const float covar_dv_acc_bias = -_params.var_acc_bias * dt2_2;
    P(IDX_DVX, IDX_ACC_BIAS_X) += covar_dv_acc_bias;
    P(IDX_DVY, IDX_ACC_BIAS_Y) += covar_dv_acc_bias;
    P(IDX_DVZ, IDX_ACC_BIAS_Z) += covar_dv_acc_bias;

    // dx row
    P(IDX_DX, IDX_DVX) += covar_dv_dp;
    P(IDX_DY, IDX_DVY) += covar_dv_dp;
    P(IDX_DZ, IDX_DVZ) += covar_dv_dp;

    const float covar_dx_acc_bias = -_params.var_acc_bias * dt3_3 * 0.5f;
    P(IDX_DX, IDX_ACC_BIAS_X) += covar_dx_acc_bias;
    P(IDX_DY, IDX_ACC_BIAS_Y) += covar_dx_acc_bias;
    P(IDX_DZ, IDX_ACC_BIAS_Z) += covar_dx_acc_bias;

    // Gyro bias row
    P(IDX_GYRO_BIAS_X, IDX_ALPHA_X) += covar_alpha_gb;
    P(IDX_GYRO_BIAS_Y, IDX_ALPHA_Y) += covar_alpha_gb;
    P(IDX_GYRO_BIAS_Z, IDX_ALPHA_Z) += covar_alpha_gb;

    const float var_gyro_bias = _params.var_gyro_bias * dt;
    P(IDX_GYRO_BIAS_X, IDX_GYRO_BIAS_X) += var_gyro_bias;
    P(IDX_GYRO_BIAS_Y, IDX_GYRO_BIAS_Y) += var_gyro_bias;
    P(IDX_GYRO_BIAS_Z, IDX_GYRO_BIAS_Z) += var_gyro_bias;

    // Accelerometer bias row
    P(IDX_ACC_BIAS_X, IDX_DVX) += covar_dv_acc_bias;
    P(IDX_ACC_BIAS_Y, IDX_DVY) += covar_dv_acc_bias;
    P(IDX_ACC_BIAS_Z, IDX_DVZ) += covar_dv_acc_bias;

    P(IDX_ACC_BIAS_X, IDX_DX) += covar_dx_acc_bias;
    P(IDX_ACC_BIAS_Y, IDX_DY) += covar_dx_acc_bias;
    P(IDX_ACC_BIAS_Z, IDX_DZ) += covar_dx_acc_bias;

    const float var_accel_bias = _params.var_acc_bias * dt;
    P(IDX_ACC_BIAS_X, IDX_ACC_BIAS_X) += var_accel_bias;
    P(IDX_ACC_BIAS_Y, IDX_ACC_BIAS_Y) += var_accel_bias;
    P(IDX_ACC_BIAS_Z, IDX_ACC_BIAS_Z) += var_accel_bias;

    // Magnetometer bias row
    const float var_mag_bias = _params.var_mag_bias * dt;
    P(IDX_MAG_BIAS_X, IDX_MAG_BIAS_X) += var_mag_bias;
    P(IDX_MAG_BIAS_Y, IDX_MAG_BIAS_Y) += var_mag_bias;
    P(IDX_MAG_BIAS_Z, IDX_MAG_BIAS_Z) += var_mag_bias;

    P(IDX_ALT_BARO, IDX_ALT_BARO) += covar_dx_acc_bias + _params.var_baro_bias * dt;
}

void MEKF::predict_limit_cov() {
    // Limit accel bias
    _kalman.x(IDX_ACC_BIAS_X) = constrain(_kalman.x(IDX_ACC_BIAS_X), -LIMIT_ACC_BIAS, LIMIT_ACC_BIAS);
    _kalman.x(IDX_ACC_BIAS_Y) = constrain(_kalman.x(IDX_ACC_BIAS_Y), -LIMIT_ACC_BIAS, LIMIT_ACC_BIAS);
    _kalman.x(IDX_ACC_BIAS_Z) = constrain(_kalman.x(IDX_ACC_BIAS_Z), -LIMIT_ACC_BIAS, LIMIT_ACC_BIAS);

    // Limit accel bias covariance
    for (int axis = 0; axis < 3; ++axis) {
        float k = _kalman.P(IDX_ACC_BIAS_X + axis, IDX_ACC_BIAS_X + axis) / VAR_LIMIT_ACC_BIAS;
        if (k > 1.0f) {
            k = sqrtf(k);
            _kalman.P.col(IDX_ACC_BIAS_X + axis) /= k;
            _kalman.P.row(IDX_ACC_BIAS_X + axis) /= k;
        }
    }

    // Limit gyro bias
    for (int axis = 0; axis < 3; ++axis) {
        _kalman.x(IDX_GYRO_BIAS_X + axis) = constrain(_kalman.x(IDX_GYRO_BIAS_X + axis),
                                                      -LIMIT_GYRO_BIAS, LIMIT_GYRO_BIAS);
    }
}

void MEKF::update_aposteriori() {
    // Update integrated estimates
    _q_estimate *= Quaternionf(1.0f,
                               _kalman.x(IDX_ALPHA_X) * 0.5f,
                               _kalman.x(IDX_ALPHA_Y) * 0.5f,
                               _kalman.x(IDX_ALPHA_Z) * 0.5f);
    _q_estimate.normalize();
    _kalman.x.block<3, 1>(IDX_ALPHA_X, 0).setZero();

    _lla_estimate += _kalman.x.block<3, 1>(IDX_DX, 0);
    _kalman.x.block<3, 1>(IDX_DX, 0).setZero();

    _vel_estimate += _kalman.x.block<3, 1>(IDX_DVX, 0);
    _kalman.x.block<3, 1>(IDX_DVX, 0).setZero();
}

Matrix3f MEKF::skew_symmetric(const Vector3f &v) {
    Matrix3f skew;
    skew << 0, -v.z(), v.y(),
            v.z(), 0, -v.x(),
            -v.y(), v.x(), 0;
    return skew;
}

template <typename T>
T MEKF::constrain(T value, T min, T max)
{
    if (value < min) {
        return min;
    } else if (value > max) {
        return max;
    } else {
        return value;
    }
}

void MEKF::add_attitude_variance(float v) {
    auto &P = _kalman.P;
    P(IDX_ALPHA_X, IDX_ALPHA_X) += v;
    P(IDX_ALPHA_Y, IDX_ALPHA_Y) += v;
    P(IDX_ALPHA_Z, IDX_ALPHA_Z) += v;
}
