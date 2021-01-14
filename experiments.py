import warnings

from process.preprocess.main import preprocess
from process.mainprocess.main import mainprocess
from process.postprocess.main import postprocess



def main():
    """データセットを使った実験
    """
    print("pre-process part".center(50, "~"))
    options, df, results_baseline, output_dir = preprocess()
    print("main-process part".center(50, "~"))
    results, fit_times = mainprocess(options, df)
    print("post-process part".center(50, "~"))
    postprocess(results, fit_times, results_baseline, output_dir, options)



if __name__ == "__main__":
    warnings.simplefilter('ignore', UserWarning)
    main()