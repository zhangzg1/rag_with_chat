import pdfplumber
from PyPDF2 import PdfReader


class DataProcess(object):

    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.data = []

    def SlidingWindow(self, sentences, kernel=512, stride=1):
        sz = len(sentences)
        cur = ""
        fast = 0
        slow = 0
        while (fast < len(sentences)):
            sentence = sentences[fast]
            if (len(cur + sentence) > kernel and (cur + sentence) not in self.data):
                self.data.append(cur + sentence + "。")
                cur = cur[len(sentences[slow] + "。"):]
                slow = slow + 1
            cur = cur + sentence + "。"
            fast = fast + 1

    '''
    pdf滑窗法解析法，把整个PDF的数据文档按句号分割，然后构建滑动窗口来存储文本数据
    '''

    def ParseAllPage(self, max_seq=512, min_len=6):
        all_content = ""
        for idx, page in enumerate(PdfReader(self.pdf_path).pages):
            page_content = ""
            text = page.extract_text()
            words = text.split("\n")
            for idx, word in enumerate(words):
                text = word.strip().strip("\n")
                if ("...................." in text or "目录" in text):
                    continue
                if (len(text) < 1):
                    continue
                if (text.isdigit()):
                    continue
                page_content = page_content + text
            if (len(page_content) < min_len):
                continue
            all_content = all_content + page_content
        sentences = all_content.split("。")
        self.SlidingWindow(sentences, kernel=max_seq)

    #  数据过滤
    def Datafilter(self, line, header, pageid, max_seq=1024):
        sz = len(line)
        if (sz < 6):
            return
        if (sz > max_seq):
            if ("■" in line):
                sentences = line.split("■")
            elif ("•" in line):
                sentences = line.split("•")
            elif ("\t" in line):
                sentences = line.split("\t")
            else:
                sentences = line.split("。")
            for subsentence in sentences:
                subsentence = subsentence.replace("\n", "")
                if (len(subsentence) < max_seq and len(subsentence) > 5):
                    subsentence = subsentence.replace(",", "").replace("\n", "").replace("\t", "")
                    if (subsentence not in self.data):
                        self.data.append(subsentence)
        else:
            line = line.replace("\n", "").replace(",", "").replace("\t", "")
            if (line not in self.data):
                self.data.append(line)

    def GetHeader(self, page):
        try:
            lines = page.extract_words()[::]
        except:
            return None
        if (len(lines) > 0):
            for line in lines:
                if ("目录" in line["text"] or ".........." in line["text"]):
                    return None
                if (line["top"] < 20 and line["top"] > 17):
                    return line["text"]
            return lines[0]["text"]
        return None

    '''
    pdf分块解析法，尽量保证一个小标题+对应文档在一个文档块，然后对每个文档块的文本内容进行提取并存储
    '''

    def ParseBlock(self, max_seq=1024):
        with pdfplumber.open(self.pdf_path) as pdf:
            for i, p in enumerate(pdf.pages):
                header = self.GetHeader(p)
                if (header == None):
                    continue
                texts = p.extract_words(use_text_flow=True, extra_attrs=["size"])[::]
                squence = ""
                lastsize = 0
                for idx, line in enumerate(texts):
                    if (idx < 1):
                        continue
                    if (idx == 1):
                        if (line["text"].isdigit()):
                            continue
                    cursize = line["size"]
                    text = line["text"]
                    if (text == "□" or text == "•"):
                        continue
                    elif (text == "警告！" or text == "注意！" or text == "说明！"):
                        if (len(squence) > 0):
                            self.Datafilter(squence, header, i, max_seq=max_seq)
                        squence = ""
                    elif (format(lastsize, ".5f") == format(cursize, ".5f")):
                        if (len(squence) > 0):
                            squence = squence + text
                        else:
                            squence = text
                    else:
                        lastsize = cursize
                        if (len(squence) < 15 and len(squence) > 0):
                            squence = squence + text
                        else:
                            if (len(squence) > 0):
                                self.Datafilter(squence, header, i, max_seq=max_seq)
                            squence = text
                if (len(squence) > 0):
                    self.Datafilter(squence, header, i, max_seq=max_seq)

    '''
    pdf非滑窗法解析法，把整个PDF的数据文档按句号分割，然后使用简单的重叠块进行数据存储
    '''

    def ParseOnePageWithRule(self, max_seq=512, min_len=6):
        for idx, page in enumerate(PdfReader(self.pdf_path).pages):
            page_content = ""
            text = page.extract_text()
            words = text.split("\n")
            for idx, word in enumerate(words):
                text = word.strip().strip("\n")
                if ("...................." in text or "目录" in text):
                    continue
                if (len(text) < 1):
                    continue
                if (text.isdigit()):
                    continue
                page_content = page_content + text
            if (len(page_content) < min_len):
                continue
            if (len(page_content) < max_seq):
                if (page_content not in self.data):
                    self.data.append(page_content)
            else:
                sentences = page_content.split("。")
                cur = ""
                for idx, sentence in enumerate(sentences):
                    if (len(cur + sentence) > max_seq and (cur + sentence) not in self.data):
                        self.data.append(cur + sentence)
                        cur = sentence
                    else:
                        cur = cur + sentence


if __name__ == "__main__":
    dp = DataProcess(pdf_path="data/car_user_manual.pdf")
    # 使用两次pdf分块解析法，分别设置最大序列长度为 1024 和 512
    dp.ParseBlock(max_seq=1024)
    dp.ParseBlock(max_seq=512)
    print(len(dp.data))
    # 使用两次pdf滑块解析法，分别设置最大序列长度为 256 和 512
    dp.ParseAllPage(max_seq=256)
    dp.ParseAllPage(max_seq=512)
    print(len(dp.data))
    # 使用两次pdf非滑块解析法，分别设置最大序列长度为 256 和 512
    dp.ParseOnePageWithRule(max_seq=256)
    dp.ParseOnePageWithRule(max_seq=512)
    print(len(dp.data))
    # 这里得到的最终文本数据是经过6次PDF文档解析得到的
    data = dp.data
    out = open("all_text.txt", "w")
    for line in data:
        line = line.strip("\n")
        out.write(line)
        out.write("\n")
    out.close()
