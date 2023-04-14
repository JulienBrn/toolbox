import cProfile
import io, pstats, re, logging
import pandas as pd
from pathlib import Path
import numpy as np

logger=logging.getLogger(__name__)

class Profile:
    def __enter__(self):
        self.pr =  cProfile.Profile()
        self.pr.enable()
        return self
    def __exit__(self, exc_type, exc_value, exc_tb):
        self.pr.disable()

    def get_results(self):
        s = io.StringIO()
        ps = pstats.Stats(self.pr, stream=s)
        ps.print_stats()
        res_str = str(s.getvalue())

        columns=["ncalls", "tottime", "percall", "cumtime", "percall2", "filename", "lineno", "function"]
        regex = re.compile(
          "\s*(?P<ncalls>\S*)\s*(?P<tottime>\S*)\s*(?P<percall>\S*)"+
          "\s*(?P<cumtime>\S*)\s*(?P<percall2>\S*)" + 
          "(("+
              "\s*(?P<filename>.*):(?P<lineno>.*)\((?P<function>.*)\)"+
          ")" 
          + "|("
          + "\s*\{(?P<function2>.*)\}"
          + "))"
            
        )
        res_list = []
        ignored_list=[]
        for i, l in enumerate(res_str.split("\n")):
            match = regex.fullmatch(l)
            if match:
              d = match.groupdict()
              if d["function2"]:
                 d["function"] = d["function2"]
                 d["filename"] = "unspecified"
                 d["lineno"] = np.nan
              del d["function2"]
              res_list.append(d)
            else:
              ignored_list.append(l)

        # logger.warning("Ignored the following lines:\n{}".format("\n\t".join(ignored_list)))
        res_df = pd.DataFrame(res_list[1:], columns = columns)
        for col in ["tottime", "percall", "cumtime", "percall2", "lineno"]:
           res_df[col] = res_df[col].astype(float)
        res_df["filename"] = res_df.apply(lambda row: Path(*Path(row["filename"]).parts[-3:]), axis=1)
        
        return res_df

