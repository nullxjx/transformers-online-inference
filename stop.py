import unittest
from transformers import StoppingCriteria
import torch
from loguru import logger


def find_shortest_prefix(suffix, chunk, stop_word):
    """
    将 suffix 和 chunk 尾部对齐，找出 suffix 中多余的字符，存放在 prefix 中
    提取出 chunk 前缀中最短的一部分 x ，使得 prefix+x = stop_word
    例如：
    suffix: abcxxabcxxx
    chunk:    cxxabcxxx
    stop_word: abc

    prefix: ab
    x: c
    """
    if len(suffix) < len(chunk):
        return ""

    prefix = ""
    for i in range(len(suffix)):
        if suffix[i:] == chunk:
            prefix = suffix[:i]
            break
    for i in range(len(chunk)+1):
        if prefix + chunk[:i] == stop_word:
            return chunk[:i]
    return ""

def remove_suffix(buffer, chunk, stop_words):
    """
        remove_suffix_completions: 移除流式补全过程中的后缀，包含stop_words或者eos之类的特殊结束符
        由于存在一个stop_word前一部分在之前一次iteration生成，后一部分在这次iteration生成，故需要缓存当前为止生成的所有字符串再进行截取判断
        :param buffer: 当前为止生成的所有字符串
        :param chunk: 当前这个iteration从TextIteratorStreamer中取到的字符串
        :param stop_words:
        :return: [移除suffix之后的word部分，是否停止]

        Examples:
           假设传进来的 words=xxxxxabcxxabcxxx, word=bcxxabcxxx, 说明上一次为止一共生成xxxxxabcxxabcxxx，这一次生成的是bcxxabcxxx
           假设有一个stop_word是abc，那么上一次没命中，abc的前面一部分a已经发出去了，这一次bcxxabcxxx还没发出去
           按照下面的逻辑，依次进行如下步骤：
             1. 从words中查找这个stop_word第一次出现的位置，然后提取这个位置之后的所有字符，存放到suffix_total里面
                比如这里，suffix_total=abcxxabcxxx
             2. 把suffix_total和word尾部对齐，找出suffix_total中前缀多余的字符，存放在prefix中，例如这里，prefix=a
                suffix_total: abcxxabcxxx
                word:          bcxxabcxxx

                注意，这里还有一种case，就是 suffix_total 比 word 短的case，例如：
                word: xxxxabcyyyy
                suffix_total: abcyyyy
                这个时候就需要直接在word中截取掉stop之后的内容，保留xxxx
             3. 提取出word前缀中最短的一部分x，使得 prefix+x = stop_word，对于这里x=bc，这个bc就是这次新生成的前缀中属于stop_word的部分
                3.1 如果x==stop_word，说明完整的stop_word都在这次生成的前缀中，这次生成的内容可以完全丢弃，不向前端返回
                3.2 如果x!=stop_word，说明stop_word中有一部前缀上次已经向前端返回了，如果这次生成的后半部分不向前端返回，
                    那么前端没办法截取掉这个多余的stop_word，所以这个也需要返回
        """
    if stop_words is None or len(stop_words) == 0:
        return chunk, False, ""
    if isinstance(stop_words, str):
        stop_words = [stop_words]
    total = buffer + chunk
    stop_words.sort(key=len, reverse=True)
    for stop_word in stop_words:
        idx = total.find(stop_word)
        if idx == -1:
            continue
        suffix = total[idx:]

        if len(suffix) < len(chunk):
            idx_ = chunk.find(stop_word)
            return chunk[:idx_], True, stop_word

        # now len(suffix) >= len(chunk)
        prefix = find_shortest_prefix(suffix, chunk, stop_word)
        if prefix == stop_word:
            return "", True, stop_word
        return prefix, True, stop_word
    return chunk, False, ""


class TestFindShortestPrefix(unittest.TestCase):
    def test_example_case_1(self):
        self.assertEqual(find_shortest_prefix("abcxxabcxxx", "cxxabcxxx", "abc"), "c")

    def test_example_case_2(self):
        self.assertEqual(find_shortest_prefix("abcxxabcxxx", "bcxxabcxxx", "abc"), "bc")

    def test_example_case_3(self):
        self.assertEqual(find_shortest_prefix("abcxxabcxxx", "abcxxabcxxx", "abc"), "abc")

    def test_example_case_4(self):
        self.assertEqual(find_shortest_prefix("<eos>", "xxx<eos>", "<eos>"), "")

    def test_example_case_5(self):
        self.assertEqual(find_shortest_prefix("abcdef", "cdef", "abcdef"), "cdef")

    def test_example_case_6(self):
        self.assertEqual(find_shortest_prefix("abcdef", "ef", "abcdef"), "ef")

    def test_example_case_7(self):
        self.assertEqual(find_shortest_prefix("abcdef", "f", "abcdef"), "f")

    def test_example_case_8(self):
        self.assertEqual(find_shortest_prefix("abcdef", "abcdef", "abcdef"), "abcdef")

    def test_example_case_9(self):
        self.assertEqual(find_shortest_prefix("aabbcc", "bbcc", "aabbcc"), "bbcc")

    def test_example_case_10(self):
        self.assertEqual(find_shortest_prefix("aabbcc", "cc", "aabbcc"), "cc")

    def test_example_case_11(self):
        self.assertEqual(find_shortest_prefix("aabbcc", "aabbcc", "aabbcc"), "aabbcc")

    def test_example_case_12(self):
        self.assertEqual(find_shortest_prefix("xyzxyz", "yzxyz", "xyz"), "yz")

    def test_example_case_13(self):
        self.assertEqual(find_shortest_prefix("xyzxyz", "zxyz", "xyz"), "z")

    def test_example_case_14(self):
        self.assertEqual(find_shortest_prefix("123123", "3123", "123"), "3")

    def test_example_case_15(self):
        self.assertEqual(find_shortest_prefix("123123", "23123", "123"), "23")

    def test_example_case_16(self):
        self.assertEqual(find_shortest_prefix("hello world", "o world", "hello"), "o")

    def test_example_case_17(self):
        self.assertEqual(find_shortest_prefix("hello world", "lo world", "hello"), "lo")

    def test_example_case_18(self):
        self.assertEqual(find_shortest_prefix("abcdefg", "cdefg", "abc"), "c")

    def test_example_case_19(self):
        self.assertEqual(find_shortest_prefix("abcdefg", "bcdefg", "abc"), "bc")

    def test_example_case_20(self):
        self.assertEqual(find_shortest_prefix("", "", ""), "")


class DefaultStopWordsCriteria(StoppingCriteria):
    def __init__(self, stops=None, encounters=1):
        super().__init__()
        if stops is None:
            stops = []
        self.stops = stops
        self.ENCOUNTERS = encounters

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


class CancelStopCriteria(StoppingCriteria):
    def __init__(self):
        super().__init__()
        self.stop = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        # 判断请求是否cancel
        if self.stop:
            logger.info("cancel model generation")
            return True
        return False

    def cancel(self):
        self.stop = True

class LogitLineStopCriteria(StoppingCriteria):
    def __init__(self, logit_first, logit_min, logit_max, k, tokenizer):
        super().__init__()

        self.logit_first = logit_first
        self.logit_min = logit_min
        self.logit_max = logit_max
        self.k = k

        self.scores = []
        self.encouter_line = 0
        self.tokenizer = tokenizer

        self.str = ""

        self.scores_line = []
        self.str_line = [""]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        self.str += self.tokenizer.decode(input_ids[0][-1])
        self.str_line[-1] += self.tokenizer.decode(input_ids[0][-1])

        self.scores.append(scores[-1][0][input_ids[0][-1]])

        if self.encouter_line == 0:
            if torch.mean(torch.tensor(self.scores)) < self.logit_first:
                return True
        else:
            if torch.mean(torch.tensor(self.scores)) < min(self.logit_min + self.k * (self.encouter_line - 1),
                                                           self.logit_max):
                return True

        if "\n" in self.tokenizer.decode(input_ids[0][-1]):
            self.scores_line.append(torch.mean(torch.tensor(self.scores)))
            self.str_line.append("")

            self.encouter_line += 1
            self.scores = []

        return False