import os
import datetime


def get_set_dir(options, args):
    """出力フォルダの定義&作成を行う

    Arguments:
        options {orderddict} -- options
        args {Object} -- コマンドライン引数

    Returns:
        string -- 出力先フォルダ
    """
    jst_zone = datetime.timezone(datetime.timedelta(hours=9), name='JST')
    dt_now = datetime.datetime.now(jst_zone)
    exp_name = options['experiments']['experiment_name']
    data_name = args.dataset
    source_name = args.source
    if source_name == 'uci':
        source_name += '/{}'.format(options['datasets']['split_feature_name'])
    output_dir = './result/{}/{}/{}/{}/'.format(exp_name, data_name, 
                    source_name, dt_now.strftime("%m%d%H%M"))

    os.makedirs(output_dir, exist_ok=True)

    return output_dir