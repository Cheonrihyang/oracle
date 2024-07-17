<%@ page contentType="text/html; charset=utf-8" pageEncoding="utf-8" %>
<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c" %>
<%@ page session="false" %>
<html>
<head>
	<title>Home</title>
	<link rel="stylesheet" type="text/css" href="/resources/css/test.css">
	<link rel="stylesheet" type="text/css" href="/resources/css/bootstrap.min.css">
	<script type="text/javascript" src="/resources/js/jquery-3.6.0.js"></script>
	<script type="text/javascript">
		$(()=>{
			$("h1").click(function (){
				$("#target").slideDown(700);
			});
			$("h1").click(function (){
				$("#target").slideUp(700);
			});
		});
	</script>
</head>
<body>
	<div class="container-md mt-5" style="background-color: skyblue">
		<h1>Hello world!</h1>

	</div>
	<div id="target" style="height: 500px; text-align: center; background-color: lightgray">
		<P>The time on the server is ${serverTime}.</P>
	</div>
</body>
</html>
