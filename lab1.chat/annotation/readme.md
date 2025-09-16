## Collecting and annoting conversations
The notebooks in this folder are used to aggregate multiple conversations and prepare these for annotation:

1. aggregating_chats.ipynb:
   - creates a single JSON with multiple conversation from a folder with conversations. It derives statistics on the speakers and their conversations.
    - preparing-annotations.ipynb: based on the student names and their persona, we create an annotation file such that three students annotate the same conversation but in different combinations.
4. aggregating_annotation.ipynb: we combine the annotations and determine the adjudicated value by majority vote or the first annotation if there is no majority (all three annoated something else).

