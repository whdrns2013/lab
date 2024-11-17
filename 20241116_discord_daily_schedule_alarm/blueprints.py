


class DateReqBody:
    def __init__(self, property_name:str, today:str):
        self.body = {
            'filter': {
                'and': [
                    {
                    'property': property_name,
                    'date': {
                        'on_or_after': today,
                        'end':True
                    }
                    }
                ],
                'or':[
                    {
                        'property': property_name,
                        'date': {
                            'equal': today                            
                        }
                    }
                ]
            }
        }