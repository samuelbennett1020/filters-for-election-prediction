from datetime import datetime


class Election:
    def __init__(self, year: str, election_date: datetime, result: dict | None):
        self.year: str = year  # poll years
        self.election_date: datetime = election_date
        self.parties: tuple = tuple(result.keys())
        self.result: dict = result


election_2024 = Election('2019_Latest', datetime(2024, 7, 12),
                         {'Lab': None, 'Con': None, 'LD': None, 'BXP/Reform': None, 'Green': None})

election_2019 = Election('2017-19', datetime(2019, 12, 12), {'Lab': 32.2, 'Con': 43.6, 'LD': 11.5})

election_2017 = Election('2015-17', datetime(2017, 6, 8), {'Lab': 40.0, 'Con': 42.3, 'LD': 7.4})

election_2015 = Election('2010-15', datetime(2015, 5, 7), {'Lab': 30.4, 'Con': 36.8, 'LD': 7.8})

election_2010 = Election('2005-10', datetime(2010, 5, 6), {'Lab': 29.0, 'Con': 36.1, 'LD': 23.0})

election_2005 = Election('2001-05', datetime(2005, 5, 5), {'Lab': 35.2, 'Con': 32.4, 'LD': 22.1})

election_2001 = Election('1997-2001', datetime(2001, 6, 7), {'Lab': 31.7, 'Con': 40.7, 'LD': 18.3})
