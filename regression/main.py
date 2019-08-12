import arguments
import cavia
import maml
import custom_cavia


if __name__ == '__main__':

    args = arguments.parse_args()

    if args.custom:
        logger = custom_cavia.run(args, log_interval=100, rerun=True)
    elif args.maml:
        logger = maml.run(args, log_interval=100, rerun=True)
    else:
        logger = cavia.run(args, log_interval=100, rerun=True)
    # else:
    #     logger = custom_cavia.run(args, log_interval=100, rerun=True)
