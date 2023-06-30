import tqdm

def patch_tqdm(update):
    class mTQDM(tqdm.tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def display(self, *args, **kwargs):
            total = self.total
            cur = self.n 
            text = self.format_meter(**{**self.format_dict, "bar_format":'{l_bar}{r_bar}'}).replace("||", "  ")
            update.emit(float(cur), float(total), str(text))
            # progress_bar.setMaximum(total)
            # progress_bar.setValue(cur)
            # progress_bar.setFormat(self.format_meter(**{**self.format_dict, "bar_format":'{l_bar}{r_bar}'}).replace("||", "  "))
            # print("MYPRINT", self.format_meter(**{**self.format_dict, "bar_format":'{l_bar}{r_bar}'}), self.format_dict, "END")
            super().display(*args, **kwargs)
            
    # setattr(module, tqdm_cls_name, mTQDM)
    return mTQDM