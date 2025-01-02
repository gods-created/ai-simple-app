from minions import (
    if_file_exists,
    training,
    predict
)
from loguru import logger

def main(*args, **kwargs) -> str:
    filedir = kwargs.get('filedir', '').strip()
    task = kwargs.get('task', 'training')
    if not if_file_exists(filedir):
        return 'The file doesn\'t exist'
    
    if task == 'training':
        response = training(filedir)
    elif task == 'predict':
        response = predict(filedir)
    else:
        response = 'The task is incorrect'

    return response


if __name__ == '__main__':
    try:
        filedir = input('The file directory with data: ')
        task = input('Current task (training, predict): ')
        main_response = main(
            filedir=filedir,
            task=task
        )
        logger.info(main_response)

    except (KeyboardInterrupt, Exception, ) as e:
        logger.error(str(e))