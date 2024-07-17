<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
	<meta charset="UTF-8">
	<title>insertBoard.jsp</title>
</head>
<body>
	<center>
		<h1>글 상세</h1>
		<a href="logout.do">로그아웃</a>
		<hr>
		<!-- 
			enctype(인코딩타입)
				이전데이터 전송 : multipart/form-data
				인코딩없이 전송 : text/plain
				디폴트: application/x-www-form-urlencoded
		 -->
		<form action="insertBoard.do" method="post" enctype="multipart/form-data">
			<table border="1">
				<tr>
					<td width="70px">제목</td>
					<td>
						<input type="text" name="title">
					</td>
				</tr>
				<tr>
					<td width="70px">작성자</td>
					<td><input type="text" name="writer"></td>
				</tr>
				<tr>
					<td width="70px">내용</td>
					<td>
						<textarea name="content" cols="40" rows="10"></textarea>
					</td>
				</tr>
				<tr>
					<td>업로드</td>
					<td>
						<input type="file" name="uploadFile">
					</td>
				</tr>
				<tr>
					<td colspan="2" align="center">
						<input type="submit" value="글 등록">
					</td>
				</tr>
			</table>
		</form>
		<br><br>
		<a href="getBoardList.do">글 목록</a>
	</center>
</body>
</html>