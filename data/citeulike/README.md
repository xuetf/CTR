Cite-U-Like dataset
-------------------

This dataset is from: http://www.cs.cmu.edu/~chongw/data/citeulike/
5-folded dataset for training and testing is also available in that link.

- users.dat
用户文章库数据，
user matrix:
  number_of_items item1 item2 ...
  
item编号从0开始

- items.dat
文章被收藏数据
item matrix:
  number_of_users user1 user2 ...

user编号从0开始

- mult.dat

文章的格式, 每篇文章unique单词数量以及每个单词对应的数量。

Under LDA, the words of each document are assumed exchangeable.  Thus,
each document is succinctly represented as a sparse vector of word
counts. The data is a file where each line is of the form:

     [M] [term_1]:[count] [term_2]:[count] ...  [term_N]:[count]

where [M] is the number of unique terms in the document, and the
[count] associated with each term is how many times that term appeared
in the document.  Note that [term_1] is an integer which indexes the
term; it is not a string.
