import sys

def is_number(data):
    try:
        if ('.' in data):
            data = float(data)
        else:
            data = int(data)
        return True

    except ValueError:
        return False


def is_date(data):
    split_data = data.split('-')
    if(len(split_data) != 3):
        return False
    else:
        for i in split_data:
            if(not is_number(i)):
                return False
        return True


def date_to_number(data):
    if(is_number(data)):
        return float(data)
    else:
        combine_num = ""
        for num in data.split('-'):
            if(not is_number(num)): sys.exit("Error: Wrong Format.")
            combine_num += num

        return int(combine_num)