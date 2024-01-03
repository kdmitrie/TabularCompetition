import enum


class TabularCompetitionDataFrameType(enum.Enum):
    ALL = 'all'
    TRAIN = 'train'
    SUBMIT = 'submit'
    ORIGINAL = 'original'

    def __lt__(self, other):
        return True


def translate(s: TabularCompetitionDataFrameType) -> str:
    translates = {TabularCompetitionDataFrameType.TRAIN: 'Train',
                  TabularCompetitionDataFrameType.SUBMIT: 'Submit',
                  TabularCompetitionDataFrameType.ORIGINAL: 'Original'}

    return translates[s] if s in translates else s
