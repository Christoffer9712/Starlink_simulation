from spacetrack import SpaceTrackClient   # pip install spacetrack
import spacetrack.operators as op

st = SpaceTrackClient("christoffer.skrap@gmail.com", "mynewpassword123")
tle_text = st.gp_history(
    object_name=op.like("STARLINK"),
    epoch=op.inclusive_range("2026-08-05", "2026-08-08"),
    orderby="norad_cat_id", format="3le")
open("starlink_3le_2026_08_08.txt", "w").write(tle_text)