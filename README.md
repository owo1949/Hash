Run training
python hit/train.py exp_name=hit_female smpl_cfg.gender=female  

Testing
python hit/train.py exp_name=hit_female smpl_cfg.gender=female  run_eval=True wdboff=True
