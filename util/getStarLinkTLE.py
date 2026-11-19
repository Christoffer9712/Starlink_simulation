from spacetrack import SpaceTrackClient   # pip install spacetrack
import spacetrack.operators as op

st = SpaceTrackClient("christoffer.skrap@gmail.com", "mynewpassword123")
tle_text = st.gp_history(
    object_name=op.like("STARLINK"),
    epoch=op.inclusive_range("2026-08-19", "2026-08-22"),
    orderby="norad_cat_id", format="3le")
open("starlink_3le.txt", "w").write(tle_text)