import untangle
import pandas as pd
import xmltodict, json
import os
from tqdm import tqdm
def parse_trials():
    path = "Data/clinicaltrials.gov-16_dec_2015/"
    df_path = "Data/Trial_Data.csv"
    all_values = []
    first_flag = True
    ## target_fields = { "clinical_study": {"brief_title": {},"official_title": {},"brief_summary":{"textblock"}, "detailed_description":{"textblock"}, "eligibility": {'criteria', 'healthy_volunteers', 'study_pop', 'maximum_age', 'minimum_age', 'gender', 'sampling_method'}}}
    for file_name in tqdm(os.listdir(path)):
        temp = {}
        if "._" not in file_name:
            file = open(path + file_name, "r")
            file_data = xmltodict.parse(file.read())
            temp['trial_id'] = file_name.split(".")[0]
            if 'brief_title' in file_data['clinical_study'].keys():
                temp['brief_title'] = file_data['clinical_study']['brief_title']
            else:
                temp['brief_title'] = None
            if 'official_title' in file_data['clinical_study'].keys():
                temp['brief_title'] = file_data['clinical_study']['official_title']
            else:
                temp['official_title'] = None
            if 'brief_summary' in file_data['clinical_study'].keys():
                temp['brief_summary'] = file_data['clinical_study']['brief_summary']['textblock']
            else:
                temp['brief_summary'] = None
            if 'detailed_description' in file_data['clinical_study'].keys():
                temp['detailed_description'] = file_data['clinical_study']['detailed_description']['textblock']
            else:
                temp['detailed_description'] = None
            eligibility_field = ['criteria', 'healthy_volunteers', 'study_pop', 'maximum_age', 'minimum_age', 'gender',
                                 'sampling_method']
            if 'eligibility' in file_data['clinical_study'].keys():
                for item in eligibility_field:
                    if item in file_data['clinical_study']['eligibility'].keys():
                        temp[item] = file_data['clinical_study']['eligibility'][item] if type(
                            file_data['clinical_study']['eligibility'][item]) is str else \
                        file_data['clinical_study']['eligibility'][item]['textblock']
                    else:
                        temp[item] = None
            else:
                for item in eligibility_field:
                    temp[item] = None

            file.close()
            all_values.append(temp)
        if len(all_values) > 10000:
            print("saving")
            if first_flag:
                df2 = pd.DataFrame(all_values)
                df2.to_csv(df_path, index=False)
                all_values = []
                first_flag = False
            else:
                df1 = pd.read_csv(df_path)
                df2 = pd.DataFrame(all_values)
                df = pd.concat([df1, df2])
                df.to_csv(df_path, index=False)
                all_values = []
    df1 = pd.read_csv(df_path)
    df2 = pd.DataFrame(all_values)
    df = pd.concat([df1, df2])
    df.to_csv(df_path, index=False)
    all_values = []
def parse_topics():
    path1 = "Data/topics-2014_2015-description.topics"
    path2 = "Data/topics-2014_2015-summary.topics"
    file1 = open(path1, "r")
    data1 = untangle.parse(file1.read())
    temp = dict()
    for item in data1.Data.TOP:
        temp[item.NUM.cdata] = {"Description": item.TITLE.cdata, "Topic_Id": item.NUM.cdata}
    file2 = open(path2, "r")
    data2 = untangle.parse(file2.read())
    for item in data2.Data.TOP:
        temp[item.NUM.cdata]["Summary"] = item.TITLE.cdata
    df = pd.DataFrame(temp.values())
    df.to_csv("Data/Filtered_Topics.csv", index=False)


if __name__ =="__main__":
    parse_topics()