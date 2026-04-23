#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace py = pybind11;

namespace {

std::array<float, 4> parse_box(const py::sequence& seq) {
    if (py::len(seq) < 4) {
        throw std::runtime_error("box must have 4 elements");
    }
    return {
        py::cast<float>(seq[0]),
        py::cast<float>(seq[1]),
        py::cast<float>(seq[2]),
        py::cast<float>(seq[3]),
    };
}

float bbox_iou_impl(const std::array<float, 4>& a, const std::array<float, 4>& b) {
    const float inter_x1 = std::max(a[0], b[0]);
    const float inter_y1 = std::max(a[1], b[1]);
    const float inter_x2 = std::min(a[2], b[2]);
    const float inter_y2 = std::min(a[3], b[3]);

    const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    const float inter_area = inter_w * inter_h;

    const float area_a = std::max(1.0f, (a[2] - a[0]) * (a[3] - a[1]));
    const float area_b = std::max(1.0f, (b[2] - b[0]) * (b[3] - b[1]));
    return inter_area / (area_a + area_b - inter_area + 1e-6f);
}

}  // namespace

float bbox_iou(const py::sequence& box_a, const py::sequence& box_b) {
    return bbox_iou_impl(parse_box(box_a), parse_box(box_b));
}

bool valid_person(
    py::array_t<float, py::array::c_style | py::array::forcecast> keypoints,
    int min_keypoints,
    float min_height
) {
    const auto buf = keypoints.request();
    if (buf.ndim != 2 || buf.shape[0] <= 0 || buf.shape[1] < 2) {
        return false;
    }

    const auto rows = static_cast<int>(buf.shape[0]);
    const auto cols = static_cast<int>(buf.shape[1]);
    const float* ptr = static_cast<float*>(buf.ptr);

    int valid_points = 0;
    float y_min = 1e9f;
    float y_max = -1e9f;

    for (int i = 0; i < rows; ++i) {
        const float x = ptr[i * cols + 0];
        const float y = ptr[i * cols + 1];
        if (x > 0.0f && y > 0.0f) {
            ++valid_points;
            y_min = std::min(y_min, y);
            y_max = std::max(y_max, y);
        }
    }

    if (valid_points < min_keypoints) {
        return false;
    }

    const float height = y_max - y_min;
    return height >= min_height;
}

bool is_duplicate_person_bbox(
    const py::sequence& box_a,
    const py::sequence& box_b,
    float frame_w,
    float frame_h,
    float duplicate_iou_th,
    float center_dx_ratio_th,
    float center_dy_ratio_th,
    float x_overlap_ratio_th,
    float vertical_gap_ratio_th,
    float split_center_dx_ratio_th,
    float contain_ratio_th
) {
    const auto a = parse_box(box_a);
    const auto b = parse_box(box_b);

    const float iou = bbox_iou_impl(a, b);
    if (iou > duplicate_iou_th) {
        return true;
    }

    const float ax1 = a[0], ay1 = a[1], ax2 = a[2], ay2 = a[3];
    const float bx1 = b[0], by1 = b[1], bx2 = b[2], by2 = b[3];

    const float aw = std::max(1.0f, ax2 - ax1);
    const float ah = std::max(1.0f, ay2 - ay1);
    const float bw = std::max(1.0f, bx2 - bx1);
    const float bh = std::max(1.0f, by2 - by1);

    const float acx = (ax1 + ax2) * 0.5f;
    const float acy = (ay1 + ay2) * 0.5f;
    const float bcx = (bx1 + bx2) * 0.5f;
    const float bcy = (by1 + by2) * 0.5f;

    if (std::abs(acx - bcx) < center_dx_ratio_th * frame_w &&
        std::abs(acy - bcy) < center_dy_ratio_th * frame_h) {
        return true;
    }

    const float x_overlap = std::max(0.0f, std::min(ax2, bx2) - std::max(ax1, bx1));
    const float x_overlap_ratio = x_overlap / std::max(1.0f, std::min(aw, bw));
    const float vertical_gap = std::max(0.0f, std::max(ay1, by1) - std::min(ay2, by2));

    if (x_overlap_ratio > x_overlap_ratio_th &&
        vertical_gap < vertical_gap_ratio_th * frame_h &&
        std::abs(acx - bcx) < split_center_dx_ratio_th * frame_w) {
        return true;
    }

    const float inner_left = std::max(ax1, bx1);
    const float inner_top = std::max(ay1, by1);
    const float inner_right = std::min(ax2, bx2);
    const float inner_bottom = std::min(ay2, by2);
    const float inner_area = std::max(0.0f, inner_right - inner_left) * std::max(0.0f, inner_bottom - inner_top);

    if (inner_area / std::max(1.0f, std::min(aw * ah, bw * bh)) > contain_ratio_th) {
        return true;
    }

    return false;
}

PYBIND11_MODULE(cpp_accel_impl, m) {
    m.doc() = "Optional C++ acceleration for high-frequency geometry and filtering in fall detection.";

    m.def("bbox_iou", &bbox_iou, "Compute IoU for two boxes");
    m.def(
        "valid_person",
        &valid_person,
        py::arg("keypoints"),
        py::arg("min_keypoints"),
        py::arg("min_height"),
        "Fast valid person check"
    );
    m.def(
        "is_duplicate_person_bbox",
        &is_duplicate_person_bbox,
        py::arg("box_a"),
        py::arg("box_b"),
        py::arg("frame_w"),
        py::arg("frame_h"),
        py::arg("duplicate_iou_th"),
        py::arg("center_dx_ratio_th"),
        py::arg("center_dy_ratio_th"),
        py::arg("x_overlap_ratio_th"),
        py::arg("vertical_gap_ratio_th"),
        py::arg("split_center_dx_ratio_th"),
        py::arg("contain_ratio_th"),
        "Fast duplicate bbox detection"
    );
}
