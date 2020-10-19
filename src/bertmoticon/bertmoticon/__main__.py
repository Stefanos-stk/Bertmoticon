import sys
import argparse
import predict 


parser = argparse.ArgumentParser()
parser.add_argument(
    '-l', '--list',  # either of this switches
    nargs='+',       # one or more parameters to this switch
    type=str,        # /parameters/ are strings
    dest='list',     # store in 'list'.
    default=[],      # since we're not specifying required.
)
parser.add_argument(
    '-g', '--guess', # either of this switches
    type=int,        # paramaters are ints
    default = 0      # default is 0
)
args = parser.parse_args()

def main():
    print("input = ", args.list, "Guesses per sentence = ",args.guess)
    pred = predict.infer_list(args.list,args.guess)
    print("predictions = ",pred)


if __name__ == "__main__":
    main()