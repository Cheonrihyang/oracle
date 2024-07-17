<%@page import="com.spring.biz.board.BoardVO"%>
<%@page import="java.util.ArrayList"%>
<%@page import="com.spring.biz.board.impl.BoardServiceImpl"%>
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
    
<%@ taglib prefix="c" uri="http://java.sun.com/jstl/core_rt" %>

<!DOCTYPE html>
<html>
<head>
	<meta charset="UTF-8">
	<title>getBoardList.jsp</title>
</head>
<body>
	<center>
		<h1>글 목록...여기는 to입니다</h1>
		<h3>관리자님 환영합니다<a href="logout.do">로그아웃</a></h3>
		<form action="getBoardList.to" method="post">
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
			<c:forEach items="${boardList }" var="board">
				<tr>
					<td>${board.seq }</td>
					<td><a href="getBoard.to?seq=${board.seq }">${board.title }</a></td>
					<td>${board.writer }</td>
					<td>${board.content }</td>
					<td>${board.regdate }</td>
					<td>${board.cnt }</td>
				</tr>
			</c:forEach>
		</table><br>
		<a href="insertBoard.to">새 글 등록</a>
	</center>
</body>
</html>