
inlets = 1;
outlets = 1;
if (jsarguments.length>1)
	myval = jsarguments[1];

var sample_dict = new Dict("sample_buffers");


function post_info()
{
	post();
	
	for (var key in sample_dict) {
	post(sample_dict[key]);
    
	}
}


function basic_getting_and_setting()
{
	// argument is the name of a dict, which may or may not already exist. 
	// in this case there is already a dict named "northern animals" and we will reference that dict.
	var d = new Dict("northern animals");
	
	// an optional 'true' arg to getnames() will get all dictionary names
	// rather than just explicitly named dictionaries
	var names = d.getnames();
	
	post("Names of existing dictionaries: " + names);
	post();	

	// getkeys() will return an array of strings, each string being a key for our dict
	var keys = d.getkeys();
	
	// access the name of a dict object as a property of the dict object
	var name = d.name;

	post_info(name, keys);
	
	// the quiet property functions the same as the @quiet attribute to dict in Max
	// it suppresses many errors or warnings if set to true
	d.quiet = false;

	// changing the name property of a dict object does _not_ rename a dict.
	// changing the name property of a dict object tells it to reference different dictionary --
 	// one that is associated with the different name.
	d.name = "lunchmeat";

	// the set() method will assign a value (2nd arg) to a key (1st arg).
	// putting the value in quotes will turn it into a symbol, otherwise it will be a float or int
	d.set("salami", "7.99");
	d.set("bologna", 1.99);

	// replace() is like set().  
	// however, if a hierarchy is specified for the key, and the intermediate tree members don't exist,
	// then the entire heirarchy will be created.
	// hierarchy is specified using double-colons as in the following example.
	d.replace("drink::hot::coffee::origin", "sumatra");

	// passing additional args to set() or replace() will assign an array as the value
	d.set("drink::hot::coffee::sizes", "small", "medium", "large");
	
	// the append() method will add values to the end of an array
	d.append("drink::hot::coffee::sizes", "mega-grande");
	
	// we can use Max dictionary syntax or JSON to set a dictionary as the value for a key using the setparse() method
	d.setparse("meatloaf", '{ "price" : "priceless", "lyric" : "I would do anything for love but I won\'t eat that" }');
	
	// or we can set the entire content of a dictionary, again with either JSON or Max dictionary syntax,
	// by simply using the parse() method
	var e = new Dict("parameter example");
	e.parse('{ "parameters" : { "param1" : { "name" : "foo", "value" : 0.1 }, "param2" : { "name" : "bar", "value" : 3.1415 } } }');

	post_info(d.name, d.getkeys());

	var price = d.get("salami");
    post("    Salami price/pound : " + price);
	post();
	
	// the contains() method can be used to see if a key exists in a dict
	// this is checking for a key that is nested in a sub-dictionary of a sub-dictionary
	var there_is_coffee = d.contains("drink::hot::coffee");
	var there_is_tea = d.contains("drink::hot::tea");
	
	if (there_is_coffee)
		post("    There is coffee!");
	if (there_is_tea)
		post("    there is tea!");
	
	var s = d.getsize("drink::hot::coffee::sizes");
	post("    Number of Coffee Sizes: " + s);
	post();
	post("    Sizes is of type " + d.gettype("drink::hot::coffee::sizes") + "   and origin is of type " + d.gettype("drink::hot::coffee::origin"));
	post();
	
	post(" ");
	post();
}


function bang()
{
	basic_getting_and_setting();
	importing_and_exporting();
	
	// additional functions available for dict:
	
	// x.clone("ark");
	// x.remove("pig");
	// x.clear();
	
}

function list(val) {
	var a = arrayfromargs(arguments);
	post("received list " + a[1] , a[2], a[3] + "\n");
	
	sample_dict.set(a[1], 'hi' );
	myval = a;
}

function view() {
	post("Info regarding the dictionary named '", sample_dict.name, "': ");
	post();
	post("    Keys: " + sample_dict.getkeys());
	post();
}