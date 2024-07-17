<%@page import="com.spring.biz.board.BoardVO"%>
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<% BoardVO board = (BoardVO)session.getAttribute("board"); %>
<!DOCTYPE html>
<html>
<head>
	<meta charset="UTF-8">
	<title>getBoard.jsp</title>
</head>
<body>
	<center>
		<h1>글 상세</h1>
		<a href="logout.do">로그아웃</a>
		<hr>
		<form action="updateBoard.do" method="post">
			<input type="hidden" name="seq" value="<%=board.getSeq() %>">
			<table border="1">
				<tr>
					<td width="70px">제목</td>
					<td>
						<input type="text" name="title" value="<%=board.getTitle() %>">
					</td>
				</tr>
				<tr>
					<td width="70px">작성자</td>
					<td><%=board.getWriter() %></td>
				</tr>
				<tr>
					<td width="70px">내용</td>
					<td>
						<textarea name="content" cols="40" rows="10"><%=board.getContent() %></textarea>
					</td>
				</tr>
				<tr>
					<td width="70px">등록일</td>
					<td><%=board.getRegdate() %></td>
				</tr>
				<tr>
					<td width="70px">조회수</td>
					<td><%=board.getCnt() %></td>
				</tr>
				<tr>
					<td colspan="2" align="center">
						<input type="submit" value="글 수정">
					</td>
				</tr>
			</table>
		</form>
		<br><br>
		<a href="insertBoard.jsp">글 등록</a>&nbsp;&nbsp;
		<a href="deleteBoard.do?seq=<%=board.getSeq() %>">글 삭제</a>&nbsp;&nbsp;
		<a href="getBoardList.do">글 목록</a>
	</center>
</body>
</html>