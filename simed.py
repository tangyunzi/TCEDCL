import os
from simcse import SimCSE
import numpy as np
import torch
import json
import random
class SimClassifier(object):
    def __init__(self):
        # Load the configuration file to obtain information about the settings
        with open("./classify_config.conf") as f:
            self.conf = eval(f.read())
        self.embedder = SimCSE(self.conf["model_name_or_path"])
        # Read sentence encoding
        sentence_list = []
        classified_sentence_path = os.path.join(self.conf["file_dir"], self.conf["classified_sentence_file"])
        with open(classified_sentence_path, 'r', encoding='utf-8') as cf:
            for line in cf:
                line = line.replace('\\', '\\\\')
                content_obj = json.loads(line)["content"][0]
                sentence = content_obj["sentence"]
                eventype = content_obj["eventype_id"]
                sentence_list.append(str(eventype) + " " + sentence)
        self.embedder.build_index(sentence_list)
        self.list_len = len(self.conf["cata_list"])
        self.matrix = np.zeros((self.list_len-1, self.list_len), dtype=int)

    def classify_query(self, query_sentence):
        """
        :param query_sentence:
        :return:
        """
        results = self.embedder.search(query_sentence, self.conf["top_k"], self.conf["threshold"])
        # Results is a tuple list, if any, unpack
        if len(results) > 0:
            final_type = str(self.list_len-1)
            final_score = 0
            for result in results:
                sentence, score = result
                type = sentence.split()[0]
                words = query_sentence.split()
                for key in self.conf["keywords"][type]:
                    if key in words:
                        score += self.conf["key_weight"]
                if score > final_score:
                    final_score = score
                    final_type = type

        else:  # If it is below the threshold, there is no matching type
            final_type = str(self.list_len-1)
            final_score = 0
        return final_type, round(final_score, 2)

    def run_classify(self):
        query_file_path = os.path.join(self.conf["file_dir"], self.conf["query_file"])
        query_result_file_path = os.path.join(self.conf["file_dir"], self.conf["query_result_file"])
        with open(query_file_path, 'r', encoding='utf-8') as qf:
            os.makedirs(os.path.dirname(query_result_file_path), exist_ok=True)
            with open(query_result_file_path, 'w', encoding='gbk') as rf:
                rf.write("Prediction type    " + "correctness    " + "Real type    " + "sentence\n")
                for line in qf:
                    line = line.replace('\\', '\\\\')
                    content_obj = json.loads(line)["content"][0]
                    sentence = content_obj["sentence"]
                    eventype = str(content_obj["eventype_id"])
                    forecast_type, score = self.classify_query(sentence)
                    if eventype == forecast_type:
                        judge = "T"
                    else:
                        judge = "F"
                    result_line = forecast_type + "    " + judge + "    "+ "    " + str(eventype) + "    " + sentence
                    rf.write(result_line + "\n")
                    self.matrix[int(eventype)][int(forecast_type)] += 1

    def matrix_eval(self):
        """
        :return:
        """
        #print(self.matrix)
        eval_result_file_path = os.path.join(self.conf["file_dir"], self.conf["eval_result_file"])
        os.makedirs(os.path.dirname(eval_result_file_path), exist_ok=True)
        with open(eval_result_file_path, 'w', encoding='gbk') as rf:
            line_data = np.sum(self.matrix, axis=1)
            column_data = np.sum(self.matrix, axis=0)
            line_data = line_data.astype(np.float64)
            column_data = column_data.astype(np.float64)
            f1 = np.zeros(self.list_len-1, np.float64)
            for i in range(self.list_len-1):
                if line_data[i] != 0:
                    line_data[i] = self.matrix[i][i] / line_data[i]
                if column_data[i] != 0:
                    column_data[i] = self.matrix[i][i] / column_data[i]
                if line_data[i] + column_data[i] != 0:
                    f1[i] = 2 * (line_data[i] * column_data[i]) / (line_data[i] + column_data[i])
                rf.write("Type" + self.conf["cata_list"][i] + ", P："+ str(column_data[i]) + ", R："  + str(line_data[i])+  ", F1：" + str(
                    f1[i]) + "\n")
            macroP = np.mean(column_data[0:self.list_len-1])
            macroR = np.mean(line_data)
            rf.write("macroP：" + str(macroP) + "\n")
            rf.write("macroR：" + str(macroR) + "\n")
            rf.write("macroF1：" + str((2 * macroP * macroR) / (macroP + macroR)) + "\n")

if __name__ == "__main__":
    sim = SimClassifier()
    sim.run_classify()
    sim.matrix_eval()