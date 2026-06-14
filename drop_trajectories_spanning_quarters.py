import pandas as pd

def last_date_in_month(year, month):
    date = pd.Timestamp(year=year, month=month, day=1)
    last_day = date + pd.offsets.MonthEnd(1)

    return last_day 

def drop_trajectories_spanning_quarters(df, last_day):
    last_valid_time_of_quarter = last_day + pd.Timedelta(hours=23)
    first_date_next_quarter = last_day + pd.Timedelta(days=1)
    print("last valid time in quarter: ", last_valid_time_of_quarter)
    print("first date next quarter: ", first_date_next_quarter)

    df["date_time_utc"] = pd.to_datetime(df["date_time_utc"])

    # Find trajectories with points in the final hour
    late_traj_ids = df.loc[
        df["date_time_utc"].between(
            last_valid_time_of_quarter,
            first_date_next_quarter,
            inclusive="left"
        ),
        "trajectory_id"
    ].unique()

    print(f"Dropping {len(late_traj_ids)} trajectories")

    return df[~df["trajectory_id"].isin(late_traj_ids)]
     

def main(version):
      
    for year in range(2023, 2023+1):
            for i in range(1, 12+1, 3):
                print(f"{year} -- quarter: {i}-{i+2}")
                df = pd.read_parquet(f"three_months/feats_new_rule_{version}/{year}_{i}_{i+2}_feats.parquet", engine="pyarrow")
                print("Dropping trajectories that have messages 1 hour before next quarter")
                traj_bef = df["trajectory_id"].nunique()
                last_day = last_date_in_month(year, month=i+2)
                df = drop_trajectories_spanning_quarters(df, last_day=last_day)
                traj_aft = df["trajectory_id"].nunique()
                print("Dropped ", traj_bef-traj_aft, " trajectories of ", traj_bef, " traj before.")
                df.to_parquet(f"three_months/feats_new_rule_{version}/{year}_{i}_{i+2}_feats.parquet", index=False)

    return

if __name__ == "__main__":
    main(version="online")