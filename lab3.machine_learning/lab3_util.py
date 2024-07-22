import pandas
import matplotlib.pyplot as plt
import seaborn as sns

# Fixing encoding problems and replacing the 'Utterance' columns with the cleaned strings
def replace_weird_tokens_in_meld(df):
    weird = ["\x92","\x97","\x91","\x93","\x94","\x85"]
    utts = []
    for utterance in df['Utterance']:
        for w in weird:
            utterance = utterance.replace(w, "'")
        utts.append(utterance)
    df['Utterance'] = utts


#### Adding proportions to plotted labels counts
def plot_labels_with_counts(labels, values):
    total = 0
    total = sum(values)
   # print('Total of values', total)
    ax = sns.barplot(x=labels, y=values)
    # Add values above bars
    for i, v in enumerate(values):
        ax.text(i, v + 0.2, str(int((v/total*100)))+'%', ha='center')
    plt.show()

