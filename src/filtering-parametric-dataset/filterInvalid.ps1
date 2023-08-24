$connected = @()
$manifold = @()
$oriented = @()
$triangular = @()

# get the path to the directory containing the meshes
$meshesDir = "./Obj_Files/Obj_Files/Artist_UVs/Cut/"

# get the path to the directory containing the executable
$executableDir = "./build/bin/invalidFilter"

$i = 0
# run the executable on each mesh file in the meshes directory
foreach ($mesh in (Get-ChildItem $meshesDir)) {
    $meshPath = $mesh
    # get the return value of the executable if the mesh file size is < 20 MB
    $output_connected = ""
    $output_manifold = ""
    $output_oriented = ""
    $output_triangular = ""
    if ((Get-Item $meshPath).length -gt 20000000) {
        continue
    }
    else
    {
        $output_connected = & $executableDir connected $meshPath
        $output_manifold = & $executableDir manifold $meshPath
        $output_oriented = & $executableDir oriented $meshPath
        $output_triangular = & $executableDir triangular $meshPath
    }

    # if the output is empty, then the executable failed to run, so such meshes will be added to the invalid meshes list, get the length of the output
    if ($output_connected.length -eq 0) {
        continue
    }
    if ($output_manifold.length -eq 0) {
        continue
    }
    if ($output_oriented.length -eq 0) {
        continue
    }
    if ($output_triangular.length -eq 0) {
        continue
    }

    # first character of the output should be 0 or 1
    if ($output_connected[0] -eq "1") {
        $connected += $meshPath
    } 
    if ($output_manifold[0] -eq "1") {
        $manifold += $meshPath
    } 
    if ($output_oriented[0] -eq "1") {
        $oriented += $meshPath
    } 
    if ($output_triangular[0] -eq "1") {
        $triangular += $meshPath
    } 

    # increment the counter
    $i++

    if ($i % 200 -eq 0) {
        Write-Host $i
        # print the count of valid and invalid meshes
        Write-Host "Not connected meshes: " $connected.Count
        Write-Host "Not manifold meshes: " $manifold.Count
        Write-Host "Not oriented meshes: " $oriented.Count
        Write-Host "Not triangular meshes: " $triangular.Count
    }
}
# print the count of valid and invalid meshes
Write-Host "Not connected meshes: " $connected.Count
Write-Host "Not manifold meshes: " $manifold.Count
Write-Host "Not oriented meshes: " $oriented.Count
Write-Host "Not triangular meshes: " $triangular.Count

# write the list of valid mesh paths to a file
$connected | Out-File "notConnectedPaths.txt"

# write the list of invalid mesh paths to a file
$manifold | Out-File "notManifoldPaths.txt"

# write the list of too large mesh paths to a file
$oriented | Out-File "notOrientedPaths.txt"

# write the list of too nice mesh paths to a file
$triangular | Out-File "notTriangularPaths.txt"
