import pandas as pd
from datetime import date

df = pd.read_csv('./wrongPredFilted.csv')


# print(df)

def year(start, end):
    age = int(end[0:4]) - int(start[0:4]) - ((int(end[4:6]), int(end[6:])) < (int(start[4:6]), int(start[6:])))
    return int(age)


def month(start, end):
    # start and end = year month day 20010131
    a = date(int(start[0:4]), int(start[4:6]), int(start[6:]))
    b = date(int(end[0:4]), int(end[4:6]), int(end[6:]))
    months = (b.year - a.year) * 12 + (b.month - a.month)
    return months


def day(start, end):
    # start and end = year month day 20010131
    day0 = date(int(start[0:4]), int(start[4:6]), int(start[6:]))
    day1 = date(int(end[0:4]), int(end[4:6]), int(end[6:]))
    days = (day1 - day0).days
    return days


studySurgeryGapYearList = []
studySurgeryGapMonthList = []
studySurgeryGapDayList = []

birthSurgeryGapYearList = []
birthSurgeryGapMonthList = []
birthSurgeryGapDayList = []

birthStudyGapYearList = []
birthStudyGapMonthList = []
birthStudyGapDayList = []

for index, row in df.iterrows():
    tmpBirthDate = row['BIRTH_DATE'].split('/')
    tmpSurgeryDate = row['SURGERY_DATE'].split('/')
    studyDate = str(row['studyDate'])
    birthDate = tmpBirthDate[2] + tmpBirthDate[0] + tmpBirthDate[1]
    surgeryDate = tmpSurgeryDate[2] + tmpSurgeryDate[0] + tmpSurgeryDate[1]

    # print(row['CLINIC'])
    # print('tmpBirthDate', tmpBirthDate)
    # print('birthDate', birthDate)
    # print('studyDate', studyDate)
    # print('surgeryDate', surgeryDate, '\n')

    studySurgeryGapYearList.append(year(studyDate, surgeryDate))
    studySurgeryGapMonthList.append(month(studyDate, surgeryDate))
    studySurgeryGapDayList.append(day(studyDate, surgeryDate))

    birthSurgeryGapYearList.append(year(birthDate, surgeryDate))
    birthSurgeryGapMonthList.append(month(birthDate, surgeryDate))
    birthSurgeryGapDayList.append(day(birthDate, surgeryDate))

    birthStudyGapYearList.append(year(birthDate, studyDate))
    birthStudyGapMonthList.append(month(birthDate, studyDate))
    birthStudyGapDayList.append(day(birthDate, studyDate))

print('stage one end')

df['studySurgeryGapYear'] = studySurgeryGapYearList
df['studySurgeryGapMonth'] = studySurgeryGapMonthList
df['studySurgeryGapDay'] = studySurgeryGapDayList

df['birthSurgeryGapYear'] = birthSurgeryGapYearList
df['birthSurgeryGapMonth'] = birthSurgeryGapMonthList
df['birthSurgeryGapDay'] = birthSurgeryGapDayList

df['birthStudyGapYear'] = birthStudyGapYearList
df['birthStudyGapMonth'] = birthStudyGapMonthList
df['birthStudyGapDay'] = birthStudyGapDayList

df.to_csv('includeGap.csv', index=False)
