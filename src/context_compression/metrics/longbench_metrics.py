import re
from fuzzywuzzy import fuzz


def compute_longbench_metric(type, predictions, references):
    total_score = 0.
    for (prediction, ground_truths) in zip(predictions, references):
        max_score = 0.
        answers = ground_truths["answers"]
        for ground_truth in answers:
            if type == "classification_score":
                # prediction = prediction.lstrip('\n').split('\n')[0]
                all_classes = ground_truths["all_classes"]
                em_match_list = []
                for class_name in all_classes:
                    if class_name in prediction:
                        em_match_list.append(class_name)
                for match_term in em_match_list:
                    if match_term in ground_truth and match_term != ground_truth:
                        em_match_list.remove(match_term)
                if ground_truth in em_match_list:
                    score = (1.0 / len(em_match_list))
                else:
                    score = 0.0
                max_score = max(score, max_score)
            elif type == "code_sim_score":
                all_lines = prediction.lstrip('\n').split('\n')
                prediction = ""
                for line in all_lines:
                    if ('`' not in line) and ('#' not in line) and ('//' not in line):
                        prediction = line
                        break
                max_score = max(fuzz.ratio(prediction, ground_truth) / 100, max_score)
            elif type == "count_score":
                numbers = re.findall(r"\d+", prediction)
                right_num = 0
                for number in numbers:
                    if str(number) == str(ground_truth):
                        right_num += 1
                final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)

                max_score = max(final_score, max_score)
            elif type == "retrieval_score":
                pattern = r'Paragraph (\d+)'
                matches = re.findall(pattern, ground_truth)
                ground_truth_id = matches[0]
                numbers = re.findall(r"\d+", prediction)
                right_num = 0
                for number in numbers:
                    if str(number) == str(ground_truth_id):
                        right_num += 1
                final_score = 0.0 if len(numbers) == 0 else right_num / len(numbers)
                max_score=max(float(final_score), max_score)
        total_score += max_score
    score = round(100 * total_score / len(predictions), 2)
    return score
