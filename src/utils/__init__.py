from datatime import datetime


def get_formatted_time():
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y/%m/%d - %H:%M:%S")
    return formatted_time
