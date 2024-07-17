<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Insert title here</title>
</head>
<body bgcolor="#ffff45" text="#000">
	<table width="100%" border="1">
		<tr>
			<td align="center" bgcolor="orange">에러 화면입니다</td>
		</tr>
	</table>
	<br>
	<table width="100%" border="1" align="center">
		<tr>
			<td align="center">
				<br><br><br>
				message: ${exception.message }
			</td>
		</tr>
	</table>
</body>
</html>