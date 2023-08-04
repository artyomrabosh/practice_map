from razdel import sentenize
import fitz
import re

ruABC = "ёйцукенгшщзхъфывапролджэячсмитьбюЁЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭЯЧСМИТЬБЮ"
enABC = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"
NUMS = r'1234567890,.-)([]:@%№$" '
acceptable_chars = ruABC + enABC + NUMS


class Parser:
    SPECIAL_CHARS = "#^&*+_=<✓α𝑎>/\≡≡Σ∑∈●}{≤≥�åðÿæπ"
    NUMBERS = ("1", "2", "3", "4", "5", "6", "7", "8", "9")
    ACCEPTABLE_CHARS = acceptable_chars
    REPLACEMENT_DICT = {
        "»": r'"',
        "«": r'"',
        "”": r'"',
        "“": r'"',
        "—": r'-',
        "–": r'-'
    }

    def __init__(self):
        pass

    @staticmethod
    def delete_repeating_whitespaces(sent: str):
        return re.sub(' +', ' ', sent)

    @staticmethod
    def delete_unicode(sent: str):
        sent = re.sub('\xad', ' ', sent)
        return sent.encode("utf-8", "ignore").decode()

    @staticmethod
    def replace_hyphenation(sent: str):
        return re.sub("(\S)- ", lambda x: x.group(1) if x.group(1) is not None else None, sent)

    def replace_chars(self, sent: str):
        for key in self.REPLACEMENT_DICT:
            sent = sent.replace(key, self.REPLACEMENT_DICT[key])
        return sent

    def mark_blocks(self, blocks):
        for block in blocks:
            if 84 < block['bbox'][0] < 86:
                block['type'] = "text"
            if block['lines'][0]['spans'][0]['font'] == 'CMUSerif-Bold':
                block['type'] = 'title'

    def blocks_to_text(self, blocks):
        textlines = []
        for block in blocks:
            block_textlines = []
            for line in block['lines']:
                for span in line['spans']:
                    block_textlines.append(span['text'])
            if not block_textlines[0].startswith("["):
                textlines += block_textlines
        return " ".join(textlines)

    def text_to_sents(self, text: str):

        sents: list[str] = [sent.text for sent in list(sentenize(text))]

        sents = list(filter(lambda x: not any(c in self.SPECIAL_CHARS for c in x), sents))
        sents = list(filter(lambda x: not x.startswith(self.NUMBERS), sents))
        sents = list(filter(lambda x: "https:" not in x, sents))
        sents = list(filter(lambda x: not re.search(r"[1-9]\.", x), sents))

        sents = [self.replace_chars(sent) for sent in sents]
        sents = [self.delete_repeating_whitespaces(sent) for sent in sents]
        sents = [self.replace_hyphenation(sent) for sent in sents]
        sents = [self.delete_unicode(sent) for sent in sents]

        sents = list(filter(lambda x: all(c in self.ACCEPTABLE_CHARS for c in x), sents))
        sents = list(filter(lambda x: len(x) > 21, sents))
        sents = list(filter(lambda x: len(x) < 512, sents))

        return sents

    def get_sentences(self, doc_path: str):
        try:
            doc = fitz.open(doc_path)
        except FileNotFoundError:
            print("File not found")
            return []

        blocks = []

        for page in doc:
            blocks += page.get_text("dict", flags=16)['blocks']

        self.mark_blocks(blocks)
        blocks = list(filter(lambda x: x['type'] == 'text', blocks))

        return self.text_to_sents(self.blocks_to_text(blocks))
