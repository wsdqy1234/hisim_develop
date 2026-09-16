from hisim.spec.accelerator.base import AcceleratorInfo


class NVIDIA:
    NVIDIA_H20 = AcceleratorInfo.from_dict(
        config={
            "name": "NVIDIA H20",
            "device_alias": ["H20", "h20_sxm"],
            "tflops": {
                "FP8_TENSOR": 296,
                "INT8_TENSOR": 296,
                "FP16_TENSOR": 148,
                "BF16_TENSOR": 148,
                "FP32": 74,
            },
            "hbm_capacity_gb": 96,
            "hbm_bandwidth_gb": 4022,
            "inter_node_bandwidth_gb": 64,
            "intra_node_bandwidth_gb": 450,
            "vendor": "NVIDIA",
            "ref": "https://viperatech.com/product/nvidia-hgx-h20",
        },
        save_to_registry=True,
    )


class ASCEND:
    ASCEND_910C = AcceleratorInfo.from_dict(
        config={
            "name": "Ascend 910C",
            "device_alias": ["910C", "910c", "ascend910c", "Ascend910C", "Ascend910"],
            "tflops": {
                "FP16_TENSOR": 320,
                "BF16_TENSOR": 320,
                "FP32": 80,
                "INT8_TENSOR": 640,
            },
            "hbm_capacity_gb": 64,
            "hbm_bandwidth_gb": 1600,
            "inter_node_bandwidth_gb": 64,
            "intra_node_bandwidth_gb": 300,
            "vendor": "Ascend",
            "ref": "Ascend 910C-A3, 16 die x 64GB HBM2e, 1600 MHz clock",
        },
        save_to_registry=True,
    )
