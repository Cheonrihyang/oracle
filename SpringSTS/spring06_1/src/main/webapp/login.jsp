<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
	<meta charset="UTF-8">
	<title>login.jsp</title>
</head>
<body>
	<center>
		<h1>로그인</h1>
		<hr>
		<form action="userCheck.do" method="get">
			<table border="1">
				<tr>
					<td bgcolor="lightgray">아이디</td>
					<td><input type="text" name="id"></td>
				</tr>
				<tr>
					<td bgcolor="lightgray">비밀번호</td>
					<td><input type="password" name="password"></td>
				</tr>
				<tr>
					<td colspan="2" align="center">
						<input type="submit" value="로그인">
					</td>
				</tr>
			</table>
		</form>
	</center>
</body>
</html>