import os

def loadConfig(args, noise_ratio):

    args.data_path = './datasets/' + args.dataset  

    if not os.path.exists(args.data_path):
        raise SystemExit(f"Error: Dataset folder {args.data_path} does not exist, please configure the dataset first!")

    # global settings
    args.warmUp = 15
    args.k = 5
    args.lamb = 0.01
    args.gamma = 10000

    args.threshold = 0.95

    if args.dataset == "UCI":
        args.batch_size = 128
        args.dims = [[6], [240], [47]]
        args.gamma = 10000
        args.k = 7
        args.start_correct = 15
        if noise_ratio > 0.35:
            args.threshold = 0.6
        else:
            args.threshold = 0.8

    elif args.dataset == "PIE":
        args.batch_size = 64
        args.dims = [[484], [256], [279]]
        args.gamma = 10000
        args.k = 5
        args.start_correct = 75
        if noise_ratio > 0.35:
            args.threshold = 0.8
        else:
            args.threshold = 0.95

    elif args.dataset == "BBC":
        args.batch_size = 128
        args.dims = [[4659], [4633], [4665], [4684]]
        args.gamma = 10000
        args.start_correct = 75
        args.epochs = 400
        if noise_ratio > 0.35:
            args.threshold = 0.8
        else:
            args.threshold = 0.95

    elif args.dataset == "Caltech101":
        args.batch_size = 128
        args.dims = [[48], [40], [254], [1984], [512], [928]]
        args.gamma = 100
        args.start_correct = 85
        if noise_ratio > 0.35:
            args.threshold = 0.8
        else:
            args.threshold = 0.95

    elif args.dataset == "Leaves":
        args.warmUp = 40
        args.lr = 0.003
        args.batch_size = 128
        args.dims = [[64], [64], [64]]
        args.gamma = 10000
        args.lamb = 0.05
        args.start_correct = 75
        if noise_ratio > 0.35:
            args.threshold = 0.8
        else:
            args.threshold = 0.95



    elif args.dataset == "CUB":
        # args.lamb = 0.0001
        # args.gamma = 1
        args.batch_size = 128
        args.dims = [[1024], [300]]
        args.start_correct = 40
        if noise_ratio > 0.35:
            args.threshold = 0.8
        else:
            args.threshold = 0.95

    args.l2 = 1
    args.lambda_epochs = args.epochs
    args.views = len(args.dims)
    args.threshold = 1 if noise_ratio < 0.1 else args.threshold

    print(args)
    return args
