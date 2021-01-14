from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap as ordereddict


CONFIG_DIR_UCI = './configs/experiments_uci.yml'
CONFIG_DIR_WEKA = './configs/experiments_weka.yml'



def get_options(args):
    """実験の設定を取得する

    Arguments:
        args {Object} -- コマンドライン引数

    Raises:
        TypeError: args.sourceが不正(weka, uci以外)の場合

    Returns:
        orderddict -- 実験の設定
    """

    config = get_configs(args, CONFIG_DIR_UCI)

    options = ordereddict()
    options['datasets'] = {}

    if args.source=='weka':
        yaml = YAML()
        with open(CONFIG_DIR_WEKA, encoding='utf-8') as f:
            config_weka = yaml.load(f)
        # sourceの情報も追加
        config_weka['datasets'][args.dataset]['source'] = 'weka'
        options['datasets'] = config_weka['datasets'][args.dataset]
    elif args.source=='uci':
        # sourceの情報も追加
        config['datasets'][args.dataset]['source'] = 'uci'
        options['datasets'] = config['datasets'][args.dataset]
    else:
        raise TypeError('args.source: "{}" is invalid.'.format(args.source))
    options['datasets'].move_to_end('source', False)
    options['experiments'] = config['experiments']
    options['model_params'] = config['model_params']
    options["experiments"]["metrics"] = config["datasets"][args.dataset]["metric"]
    # # metrics情報の追加
    # metrics = args.metrics
    # if metrics is None:
    #     task = options["datasets"]["task"]
    #     if task=="regression":
    #         metrics = "RMSE"
    #     elif task=="classification":
    #         metrics = "error rate"
    # options["experiments"]["metrics"] = metrics
    options['experiments'].move_to_end('methods', True)

    return options


def get_configs(args, config_dir):
    """ymlファイルからconfig情報を取得する

    Arguments:
        args {Object} -- コマンドライン引数
        config_dir {string} -- ymlがあるディレクトリ

    Returns:
        orderddict -- ymlに記述されたconfig情報
    """
    
    yaml = YAML()
    with open(config_dir, encoding='utf-8') as f:
        config = yaml.load(f)

    return config



def set_options(options, output_dir):
    """実験の設定を保存する

    Arguments:
        options {orderddict} -- config
        output_dir {string} -- 出力ディレクトリ

    Returns:
        [type] -- [description]
    """
    
    output_file = output_dir + "options.yml"

    yaml = YAML()
    with open(output_file, 'w') as f:
        yaml.dump(options, stream=f)

    return options