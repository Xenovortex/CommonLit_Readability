import argparse

if __name__ == "__main__":
    # parser
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", dest="mode", action="store", 
                        help="Choose execution mode: 'create_features', 'cross_valid', 'train'")

    parser.set_defaults(
        mode=None
    )

    args = parser.parse_args()

    if args.mode is None:
        raise argparse.ArgumentError("No argument provided for mode")

    elif args.mode == 'create_features':
        pass

    elif args.mode == 'cross_valid':
        pass

    elif args.mode == 'train':
        pass
