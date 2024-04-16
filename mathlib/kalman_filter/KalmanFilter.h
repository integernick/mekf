#pragma once

#include <Eigen/Core>

/**
 * @addtogroup kalman_filter_module
 * @{ 
 */

template<int X_SIZE>
class KalmanFilter {
public:
    typedef Eigen::Matrix<float, X_SIZE, 1> VectorX;
    typedef Eigen::Matrix<float, X_SIZE, X_SIZE> MatrixP;

    VectorX x;
    MatrixP P;

    KalmanFilter() {
        x.setZero();
        P.setZero();
    }

    void reset(size_t idx, float value, float var) {
        x(idx) = value;
        P(idx, idx) = var;
        for (size_t j = 0; j < X_SIZE; ++j) {
            if (j != idx) {
                P(idx, j) = 0.0f;
                P(j, idx) = 0.0f;
            }
        }
    }

    template<typename F_FUNC, typename Q_FUNC>
    void predict_cov(F_FUNC f, Q_FUNC q) {
        for (size_t i = 0; i < X_SIZE; ++i) {
            f(P.col(i));
        }
        for (size_t i = 0; i < X_SIZE; ++i) {
            f(P.row(i));
        }
        q(P);
    }

    template<int Z_SIZE, typename H_FUNC>
    void correct_full(H_FUNC h, const Eigen::Matrix<float, Z_SIZE, X_SIZE> &H, const Eigen::Matrix<float, Z_SIZE, 1> &z,
                      const Eigen::Matrix<float, Z_SIZE, Z_SIZE> &R) {
        Eigen::Matrix<float, Z_SIZE, 1> y = z - h(x);
        Eigen::Matrix<float, Z_SIZE, Z_SIZE> S = H * P * H.transpose() + R;
        Eigen::Matrix<float, X_SIZE, Z_SIZE> K = P * H.transpose() * S.inverse();
        x += K * y;
        P -= K * H * P;
    }

    template<int Z_SIZE, int BLOCK_SIZE>
    void correct_block_err(int block_offs, const Eigen::Matrix<float, Z_SIZE, 1> &err,
                           const Eigen::Matrix<float, Z_SIZE, BLOCK_SIZE> &H,
                           const Eigen::Matrix<float, Z_SIZE, Z_SIZE> &R) {
        Eigen::Matrix<float, Z_SIZE, Z_SIZE> S = H * P.block(block_offs, block_offs, BLOCK_SIZE, BLOCK_SIZE) * H.transpose() + R;
        Eigen::Matrix<float, X_SIZE, Z_SIZE> K = P.block(0, block_offs, X_SIZE, BLOCK_SIZE) * H.transpose() * S.inverse();
        x += K * err;
        P -= K * (H * P.block(block_offs, 0, BLOCK_SIZE, X_SIZE));
    }

    void correct(size_t idx, float value, float var) {
        float y = value - x(idx);
        float S = P(idx, idx) + var;
        Eigen::Matrix<float, X_SIZE, 1> K = P.col(idx) / S;
        x += y * K;
        P -= K * P.row(idx);
    }
};

/**
 * @}
 */
