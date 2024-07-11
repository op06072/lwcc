from lwcc import LWCC
import argparse

MODEL_NAMES = [
    "CSRNet",
    "Bay",
    "DM-Count",
    "SFANet",
]

MODEL_WEIGHTS = ["SHA", "SHB", "QNRF"]


def main():
    parser = argparse.ArgumentParser(description="Cli for using lwcc")

    parser.add_argument(
        "model", choices=MODEL_NAMES, type=str, help="model used"
    )
    parser.add_argument("image", type=str, help="path to image")
    parser.add_argument(
        "-w",
        "--weights",
        choices=MODEL_WEIGHTS,
        type=str,
        default=MODEL_WEIGHTS[0],
        help="model weights (Shanghai A/B, QNRF), default SHA",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        default=False,
        help="do not down-scale image",
    )
    parser.add_argument(
        "-g",
        "--gray",
        action="store_true",
        default=False,
        help="image is gray-scale",
    )

    args = parser.parse_args()
    resize_img = not args.no_resize

    print(
        LWCC.get_count(
            args.image,
            model_name=args.model,
            model_weights=args.weights,
            resize_img=resize_img,
            is_gray=args.gray,
        )
    )


if __name__ == "__main__":
    main()
