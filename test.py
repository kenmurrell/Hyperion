import csv
import prep
from tqdm import tqdm
import formatter as fmt


# with open("./datasets/RottenTomatoesSentiments-1.csv", "r",encoding='UTF-8') as dataset1, open("./datasets/RottenTomatoesSentiments.csv",'w',newline='') as dataset2:
#     reader = csv.reader(dataset1, delimiter=' ')
#     writer = csv.writer(dataset2, delimiter=',')
#
#     for row in reader:
#         if row:
#             writer.writerow(row)


MAIN_DATASET_PATH = "./datasets/RottenTomatoesSentiments.csv"
# POS_DATASET_PATH = './datasets/tw-data.pos'
# NEG_DATASET_PATH = './datasets/tw-data.neg'
POS_DATASET_PATH = './datasets/rt-polarity.pos'
NEG_DATASET_PATH = './datasets/rt-polarity.neg'


#
# for line in tqdm(csv.reader(lines), total=len(lines)):
#     tweet = line[1].strip()
#     tweet = fmt.all(tweet)
#     if line[0].strip() == '1':
#         try:
#             pos_dataset.write(tweet.strip()+'\n')
#         except UnicodeEncodeError:
#             e_p+=1
#     else:
#         try:
#             neg_dataset.write(tweet.strip()+'\n')
#         except UnicodeEncodeError:
#             e_n+=1
# print("DONE\nP errors: "+str(e_p)+"\nN errors: "+str(e_n))
