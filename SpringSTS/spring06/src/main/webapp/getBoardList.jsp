<%@page import="com.spring.biz.board.BoardVO"%>
<%@page import="java.util.ArrayList"%>
<%@page import="com.spring.biz.board.impl.BoardServiceImpl"%>
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<%
	ArrayList<BoardVO> boardList = (ArrayList<BoardVO>)session.getAttribute("boardList");
%>
<!DOCTYPE html>
<html>
<head>
	<meta charset="UTF-8">
	<title>getBoardList.jsp</title>
</head>
<body>
	<center>
		<h1>글 목록</h1>
		<h3>관리자님 환영합니다<a href="logout.do">로그아웃</a></h3>
		<form action="getBoardList.jsp" method="post">
			<table border="1" width="700px">
				<tr>
					<td align="center">
						<select name="searchCondition">
							<option value="TITLE">제목
							<option value="CONTENT">내용
						</select>
						<input name="searchKeyword" type="text">
						<input type="submit" value="검색">
					</td>
				</tr>
			</table>
		</form>
		<table border="1" width="700px">
			<tr>
				<th width="100px">번호</th>
				<th width="200px">제목</th>
				<th width="100px">작성자<htd>
				<th width="350px">내용</th>
				<th width="150px">등록일</th>
				<th width="100px">조회수</th>
			</tr>
			<%
				for (BoardVO board : boardList){
			%>
			<tr>
				<td><%=board.getSeq() %></td>
				<td><a href="getBoard.do?seq=<%=board.getSeq()%>"><%=board.getTitle() %></a></td>
				<td><%=board.getWriter() %></td>
				<td><%=board.getContent() %></td>
				<td><%=board.getRegdate() %></td>
				<td><%=board.getCnt() %></td>
			</tr>
			<%
				}
			%>
		</table><br>
		<a href="insertBoard.jsp">새 글 등록</a>
	</center>
</body>
</html>