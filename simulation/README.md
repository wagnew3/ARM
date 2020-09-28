# TableTop Simulator Details

Here, we describe the details of how the data is generated for the tabletop scenarios.

## External data used

* ShapeNetCore v2 (link website)
* SUNCG Houses/Rooms (link website)

## Some introductory notes

This code is based on the pybullet_suncg public codebase by msavva (link website). It extends (not in the Python sense) the main class, pybullet\_suncg.simulator.BulletSimulator. We use the +Y axis as the up axis.

## Simulation Pipeline Details

* Load a random house and room from SUNCG
	* Select a random house given house\_ids
	* Select a random room in that house that satisfies some basic properties: 
		* minimum size
		* valid room type (e.g. kitchen, office)
	* Filter room objects of certain types. Don't load them. This is a manually specified list
		* People, desks, tables, chairs, etc.
	* Don't load any room object that is on top of a filtered object. E.g. don't load plates/cups if they are on top of a table that was filtered out.
* Load a random ShapeNet table
	* Select a random table given table_ids. 
		* These have been filtered: 
			* The tabletop (in x,z dimensions only) is approximately the size of the entire bounding box
			* The tabletop vertices must be a convex shape
			* The table must have an accompanying texture (for visualization purposes only)
			* Corner tables and weirdly shaped tables are filtered out. I'm mainly looking for simple tabletops.
	* Scale the table so it's between 0.75 and 1 meters tall.
	* Sample the table so that it's not in collision with room walls, or any objects in the room.
* Load random ShapeNet objects
	* Select a random number of ShapeNet objects to generate on the table
	* Only use certain ShapeNet objects. This is a manually specified list
		* Bowls, cups, plates, laptops, caps, etc.
	* For each sampled object
		* Scale it so it's not too big w.r.t. the table
		* Either sample it in a standard orientation (straight up), or a randomly chosen orientation (with some height off the table)
		* Make sure it's not in collision with other objects
	* Simulate the scene until the objects come to equilibrium.
* Remove objects that have fallen off the table from the scene.
* Sample camera views
	* Sample camera position:
		* Pick a point on the xz bounding box of the table, randomly choose offset from that point. 
		* Sample y distance from tabletop height
	* Generate RGB/Depth/Segmentation from camera position looking at tabletop center. 
	* Camera parameters are specified in terms of
		* FOV
		* near/far planes
		* aspect ratio
		
