"""data_preprocessor

Makes a data suitable for spam_detector classifier. This tool extracts features such
as 'phone number validity' and 'name validity'. Also, this tool removes redundant data
that is not used by classifier.
"""

import pandas as pd
import string
import re


names_data = pd.read_csv('https://raw.githubusercontent.com/Gzaborey/email_spam/main/names_dataframe.csv',
                         index_col='Unnamed: 0')
names_list = [str(name).lower() for name in names_data.iloc[:, -1]]


def validate_name(string_to_validate, name_data):
    """Checks if given name exists in Database."""

    for word in string_to_validate.lower().split():
        if word in name_data:
            name_validity = 1
            break
    else:
        name_validity = 0
    return name_validity


def validate_phone_number(phone_number):
    """Checks if given phone number is possible."""

    temp1 = [char for char in phone_number if (char not in string.punctuation and char != ' ')]
    temp2 = ''.join(temp1)
    if len(temp2) > 12 or len(temp2) < 9:
        return 0
    else:
        for char in range(7):
            temp1.pop()
        temp2 = ''.join(temp1)
        pattern1 = re.compile(r'(380|0|)')
        first_condition = temp2[-3::-1][::-1] in pattern1.findall(temp2[-3::-1][::-1])
        pattern2 = re.compile(r'(39|67|68|96|97|98|50|66|95|99|63|93|91|92|94)')
        second_condition = temp2[-1:-3:-1][::-1] in pattern2.findall(temp2[-1:-3:-1][::-1])
        if first_condition and second_condition:
            return 1
        else:
            return 0


def preprocess(email_data):
    """Takes Pandas Dataframe as input. Preprocesses data to be suitable for spam_detector classifier."""

    email_data = email_data.drop(['viber', 'telegram', 'datetime', 'age', 'Unnamed: 0'], axis=1, errors='ignore')
    email_data = email_data.drop_duplicates()
    email_data = email_data.reset_index(drop=True)
    email_data = email_data.rename(columns={'add': 'message'})
    email_data['message'] = email_data['message'].astype('str')

    email_data = email_data.fillna('')
    email_data['message'] = email_data['message'].replace('nan', '')

    email_data.insert(loc=3, column='email_domain',
                      value=email_data['email'].apply(lambda x: x.lower().split('@')[-1].strip()))
    email_data.insert(loc=1, column='valid_first_name',
                      value=email_data.name.astype('str').apply(lambda x: validate_name(x, names_list)))
    email_data.insert(loc=3, column='valid_phone_number',
                      value=email_data.phone.astype('str').apply(lambda x: validate_phone_number(x)))

    return email_data
