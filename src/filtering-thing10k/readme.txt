categorized alll 10K meshes in Thingi into the following:

-very Bad Mesh Paths (913 files): These are triangle meshes that have > 5% of their faces violating the inequality (with delta = 10^-3 * avgEdge Length). Also all of them are -manifold, oriented, connected and < 20 MBs
-kinda Bad Mesh Paths (2849 files): These are triangle meshes that have some faces violating the inequality (but less than 5%). Also all of them are manifold, oriented, connected and < 20 MBs
-too Nice Mesh Paths (728 files): These are triangle meshes that have No faces violating the inequality . Also all of them are manifold, oriented, connected and < 20 MBs
-Invalid Mesh Paths (5053 files): These are meshes that aren't easy to work with. i.e. Non-manifold, disconnected, non-oriented, not triangle or couldn't be loaded at all.
-too Large Mesh Paths (457 files): These files are > 20 MBs so I didn't parse them intentionally to save time. (We can do that later if needed)

These files ids are stored in a .txt file for each category, So one can easily take the id from there and get it from https://ten-thousand-models.appspot.com/