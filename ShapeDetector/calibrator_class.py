class Calibrator:
    low_threshold = 0
    low_max = 0
    low_min = 0
    high_threshold = 0
    high_max = 0
    high_min = 0
    
    def __new__(cls,*args,**kwargs):
        return super().__new__(cls)
    
    def __init__(self, _low_threshold, _high_threshold, 
                 _low_max, _low_min,
                 _high_max, _high_min):
        Calibrator.low_max = _low_max
        Calibrator.low_min = _low_min
        Calibrator.low_threshold = _low_threshold
        Calibrator.high_min = _high_min
        Calibrator.high_max = _high_max
        Calibrator.high_threshold = _high_threshold
    
    @classmethod
    def LowChange(cls, new_value):
        # assert new_value.isdigit()
        _low_min = getattr(cls, "low_min")
        _low_max = getattr(cls, "low_max")
        if cls.low_min <= new_value <= cls.low_max:
            setattr(cls, "low_threshold", new_value)
            # low_threshold = new_value
        elif _low_min <= new_value <= _low_max:
            setattr(cls, "low_threshold", new_value)
            # low_threshold = new_value
        else:
            Exception
       
    @classmethod 
    def HighChange(cls,new_value):
        # assert new_value.isdigit()
        # _high_min = getattr(cls, "high_min")
        # _high_max = getattr(cls, "high_max")
        if cls.high_min <= new_value <= cls.high_max:
            setattr(cls, "high_threshold", new_value)
            # high_threshold = new_value
        # elif _high_min <= new_value <= _high_max:
        #     setattr(cls, "high_threshold", new_value)
        #     # high_threshold = new_value
        else:
            Exception